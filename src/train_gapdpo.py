import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from adv_trainers import trainer_dpo2 as trainer_impl
from adv_trainers.data import Loader
from dpo_common import PairwiseDPOTrainerBase
from chosen4_data import (
    HarmBenchChosen4Safety,
    SafetyEnvelopeDatacollator,
)
from train_sft import (
    build_logger,
    cleanup_distributed,
    get_model,
    get_rank,
    is_main_process,
    maybe_wrap_ddp,
    replace_loaders_for_ddp,
    scale_batch_sizes_for_ddp,
    setup_distributed,
)


def parse_indices(value: str):
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise argparse.ArgumentTypeError("--chosen-indices must contain at least one index")
    return indices


def indices_tag(indices):
    return "".join(str(index) for index in indices)


@dataclass
class AdvTrainerConfig(trainer_impl.AdvTrainerConfig):
    chosen_indices: tuple = field(default_factory=lambda: (0, 1, 2, 3))
    num_harmful_samples: int = 4
    listwise_reduce: str = "mean"
    harmful_reduce: str = "mean"
    listwise_temperature: float = 1.0
    chosen_consistency_alpha: float = 0.1
    harmful_consistency_alpha: float = 0.0
    wandb_run_name: str = None


class AdvTrainerChosenConsistencyHarmfulDPO(PairwiseDPOTrainerBase):
    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.chosen_indices = tuple(trainer_config.chosen_indices)
        self.num_harmful_samples = trainer_config.num_harmful_samples
        self.listwise_reduce = trainer_config.listwise_reduce
        self.harmful_reduce = trainer_config.harmful_reduce
        self.listwise_temperature = trainer_config.listwise_temperature
        self.chosen_consistency_alpha = trainer_config.chosen_consistency_alpha
        self.harmful_consistency_alpha = trainer_config.harmful_consistency_alpha
        self.wandb_run_name = trainer_config.wandb_run_name
        super().__init__(trainer_config=trainer_config, **kwargs)

        self.train_datacollator = SafetyEnvelopeDatacollator(
            tokenizer=self.tokenizer,
            name=trainer_config.datacollator_config.name,
            chosen_indices=self.chosen_indices,
            num_harmful_samples=self.num_harmful_samples,
        )
        self.trainloader = Loader(
            dataset=self.trainset,
            batch_size=trainer_config.batch_size,
            train=True,
            collate_fn=self.train_datacollator,
            num_workers=0,
        )

        # AdvTrainerBase can miss PEFT adapters after DDP wrapping. Repair the
        # reference-model pointer so disable_adapter() is used under torchrun.
        if self._inner_peft_model is None and hasattr(self.model, "_unwrap"):
            inner_model = self.model._unwrap()
            if hasattr(inner_model, "disable_adapter"):
                self._inner_peft_model = inner_model
                if self.logger is not None:
                    self.logger.info("Use DDP-unwrapped adapter-disabled policy as DPO reference model.")

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or (
                f"qwen_dpo_260618_chosen_consistency_idx{indices_tag(self.chosen_indices)}_"
                f"h{self.num_harmful_samples}_beta{self.dpo_beta}"
            )
            self.wandb_run.config.update(
                {
                    "loss_type": "chosen_consistency_harmful_dpo",
                    "chosen_indices": self.chosen_indices,
                    "num_harmful_samples": self.num_harmful_samples,
                    "listwise_reduce": self.listwise_reduce,
                    "harmful_reduce": self.harmful_reduce,
                    "listwise_temperature": self.listwise_temperature,
                    "chosen_consistency_alpha": self.chosen_consistency_alpha,
                    "harmful_consistency_alpha": self.harmful_consistency_alpha,
                },
                allow_val_change=True,
            )

        if self.logger is not None:
            self.logger.info(
                "Chosen-Consistency Harmful DPO enabled: chosen_indices=%s, num_harmful=%d, "
                "chosen_reduce=%s, harmful_reduce=%s, consistency_alpha=%.4f, dpo_beta=%.4f",
                self.chosen_indices,
                self.num_harmful_samples,
                self.listwise_reduce,
                self.harmful_reduce,
                self.chosen_consistency_alpha,
                self.dpo_beta,
            )
            self.logger.info(
                "Symmetric consistency enabled: chosen_alpha=%.4f, harmful_alpha=%.4f, "
                "chosen_count=%d, harmful_count=%d",
                self.chosen_consistency_alpha,
                self.harmful_consistency_alpha,
                len(self.chosen_indices),
                self.num_harmful_samples,
            )

    def _aggregate_rewards(self, rewards, reduce):
        if reduce == "mean":
            return rewards.mean(dim=1)
        if reduce == "min":
            return rewards.min(dim=1).values
        if reduce == "max":
            return rewards.max(dim=1).values
        if reduce == "logsumexp":
            temp = self.listwise_temperature
            if temp <= 0:
                raise ValueError(f"listwise_temperature must be positive, got {temp}")
            return temp * torch.logsumexp(rewards / temp, dim=1) - temp * np.log(rewards.shape[1])
        raise ValueError(f"Unsupported reward reduce={reduce}")

    def _compute_chosen_consistency_terms(self, inputs, advsfx_inputs):
        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        chosen_ids = inputs["chosen_ids"]
        chosen_mask = inputs["chosen_mask"]
        harmful_ids = inputs["harmful_ids"]
        harmful_mask = inputs["harmful_mask"]
        bs, num_chosen, chosen_len = chosen_ids.shape
        _, num_harmful, harmful_len = harmful_ids.shape

        chosen_context = {
            "context_ids": context_inputs["context_ids"].repeat_interleave(num_chosen, dim=0),
            "context_mask": context_inputs["context_mask"].repeat_interleave(num_chosen, dim=0),
        }
        harmful_context = {
            "context_ids": context_inputs["context_ids"].repeat_interleave(num_harmful, dim=0),
            "context_mask": context_inputs["context_mask"].repeat_interleave(num_harmful, dim=0),
        }

        chosen_ids_flat = chosen_ids.reshape(bs * num_chosen, chosen_len)
        chosen_mask_flat = chosen_mask.reshape(bs * num_chosen, chosen_len)
        harmful_ids_flat = harmful_ids.reshape(bs * num_harmful, harmful_len)
        harmful_mask_flat = harmful_mask.reshape(bs * num_harmful, harmful_len)

        pi_chosen_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=chosen_context,
            response_ids=chosen_ids_flat,
            response_mask=chosen_mask_flat,
        ).reshape(bs, num_chosen)
        pi_harmful_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=harmful_context,
            response_ids=harmful_ids_flat,
            response_mask=harmful_mask_flat,
        ).reshape(bs, num_harmful)

        with torch.no_grad():
            ref_chosen_logp = self._compute_reference_response_logp(
                context_inputs=chosen_context,
                response_ids=chosen_ids_flat,
                response_mask=chosen_mask_flat,
            ).reshape(bs, num_chosen)
            ref_harmful_logp = self._compute_reference_response_logp(
                context_inputs=harmful_context,
                response_ids=harmful_ids_flat,
                response_mask=harmful_mask_flat,
            ).reshape(bs, num_harmful)

        reward_chosen = self.dpo_beta * (pi_chosen_logp - ref_chosen_logp)
        reward_harmful = self.dpo_beta * (pi_harmful_logp - ref_harmful_logp)
        aggregate_reward_chosen = self._aggregate_rewards(reward_chosen, self.listwise_reduce)
        aggregate_reward_harmful = self._aggregate_rewards(reward_harmful, self.harmful_reduce)

        delta_pi = self._aggregate_rewards(pi_chosen_logp, self.listwise_reduce) - self._aggregate_rewards(
            pi_harmful_logp,
            self.harmful_reduce,
        )
        delta_ref = self._aggregate_rewards(ref_chosen_logp, self.listwise_reduce) - self._aggregate_rewards(
            ref_harmful_logp,
            self.harmful_reduce,
        )

        return {
            "pi_chosen_logp": pi_chosen_logp,
            "pi_harmful_logp": pi_harmful_logp,
            "ref_chosen_logp": ref_chosen_logp,
            "ref_harmful_logp": ref_harmful_logp,
            "delta_pi": delta_pi,
            "delta_ref": delta_ref,
            "reward_chosen": reward_chosen,
            "reward_harmful": reward_harmful,
            "aggregate_reward_chosen": aggregate_reward_chosen,
            "aggregate_reward_harmful": aggregate_reward_harmful,
            "reward_margin": aggregate_reward_chosen - aggregate_reward_harmful,
        }

    def _build_attack_target_inputs(self, inputs):
        harmful_ids = inputs["harmful_ids"]
        harmful_mask = inputs["harmful_mask"]

        if harmful_ids.dim() == 2:
            return harmful_ids, harmful_mask
        if harmful_ids.dim() != 3:
            raise ValueError(f"Expected harmful_ids to be 2D or 3D, got shape={tuple(harmful_ids.shape)}")

        return harmful_ids[:, 0, :].contiguous(), harmful_mask[:, 0, :].contiguous()

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps
        terms = self._compute_chosen_consistency_terms(inputs=inputs, advsfx_inputs=advsfx_inputs)

        dpo_loss = -F.logsigmoid(terms["reward_margin"]).mean()
        chosen_consistency_loss = terms["reward_chosen"].var(dim=1, unbiased=False).mean()
        harmful_consistency_loss = terms["reward_harmful"].var(dim=1, unbiased=False).mean()
        align_loss = (
            dpo_loss
            + (self.chosen_consistency_alpha * chosen_consistency_loss)
            + (self.harmful_consistency_alpha * harmful_consistency_loss)
        )
        align_alpha_loss = align_loss * self.adv_alpha
        (align_alpha_loss * scale).backward()

        utility_loss, utility_alpha_loss = self._backward_utility_loss(utility_inputs, scale)
        reward_chosen = terms["reward_chosen"]
        reward_harmful = terms["reward_harmful"]
        aggregate_reward_chosen = terms["aggregate_reward_chosen"]
        aggregate_reward_harmful = terms["aggregate_reward_harmful"]
        pairwise_win_rate = (
            reward_chosen.unsqueeze(2) > reward_harmful.unsqueeze(1)
        ).to(torch.float32).mean().item()

        return {
            "align_loss": align_loss.item(),
            "align_alpha_loss": align_alpha_loss.item(),
            "dpo_loss": dpo_loss.item(),
            "chosen_consistency_loss": chosen_consistency_loss.item(),
            "chosen_consistency_alpha_loss": (
                self.chosen_consistency_alpha * chosen_consistency_loss
            ).item(),
            "harmful_consistency_loss": harmful_consistency_loss.item(),
            "harmful_consistency_alpha_loss": (
                self.harmful_consistency_alpha * harmful_consistency_loss
            ).item(),
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss.item() + utility_alpha_loss,
            "listwise_dpo_acc": (
                aggregate_reward_chosen > aggregate_reward_harmful
            ).to(torch.float32).mean().item(),
            "pairwise_win_rate": pairwise_win_rate,
            "dpo_margin": (terms["delta_pi"] - terms["delta_ref"]).mean().item(),
            "dpo_acc": (terms["delta_pi"] > terms["delta_ref"]).to(torch.float32).mean().item(),
            "reward_chosen": reward_chosen.mean().item(),
            "reward_chosen_min": reward_chosen.min(dim=1).values.mean().item(),
            "reward_chosen_max": reward_chosen.max(dim=1).values.mean().item(),
            "reward_harmful": reward_harmful.mean().item(),
            "reward_harmful_min": reward_harmful.min(dim=1).values.mean().item(),
            "reward_harmful_max": reward_harmful.max(dim=1).values.mean().item(),
            "aggregate_reward_chosen": aggregate_reward_chosen.mean().item(),
            "aggregate_reward_harmful": aggregate_reward_harmful.mean().item(),
            "reward_margin": terms["reward_margin"].mean().item(),
            "pi_logp_chosen": terms["pi_chosen_logp"].mean().item(),
            "pi_logp_harmful": terms["pi_harmful_logp"].mean().item(),
            "ref_logp_chosen": terms["ref_chosen_logp"].mean().item(),
            "ref_logp_harmful": terms["ref_harmful_logp"].mean().item(),
        }


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)
    utils.add_train_args(parser)
    parser.add_argument("--chosen-indices", type=parse_indices, default=(0, 1, 2, 3))
    parser.add_argument("--num-harmful-samples", type=int, default=4)
    parser.add_argument("--listwise-reduce", choices=["mean", "min", "logsumexp"], default="mean")
    parser.add_argument("--harmful-reduce", choices=["mean", "min", "max", "logsumexp"], default="mean")
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument("--chosen-consistency-alpha", type=float, default=0.1)
    parser.add_argument("--harmful-consistency-alpha", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def get_trainset(args, logger):
    if args.dataset != "harmbench-chosen4":
        raise ValueError("Chosen-Consistency Harmful DPO requires --dataset harmbench-chosen4")
    if args.num_harmful_samples < 1:
        raise ValueError(f"--num-harmful-samples must be positive, got {args.num_harmful_samples}")
    if args.chosen_consistency_alpha < 0:
        raise ValueError(
            f"--chosen-consistency-alpha must be non-negative, got {args.chosen_consistency_alpha}"
        )
    if args.harmful_consistency_alpha < 0:
        raise ValueError(
            f"--harmful-consistency-alpha must be non-negative, got {args.harmful_consistency_alpha}"
        )

    trainset = HarmBenchChosen4Safety()
    for behavior_id, chosen_list in trainset.chosen_ds.items():
        if max(args.chosen_indices) >= len(chosen_list):
            raise ValueError(
                f"behavior_id={behavior_id} has only {len(chosen_list)} chosen responses; "
                f"cannot use chosen_indices={args.chosen_indices}"
            )
        if len(trainset.target_ds[behavior_id]) < 1:
            raise ValueError(f"behavior_id={behavior_id} has no harmful targets")

    if is_main_process():
        logger.info(
            "Using chosen-consistency chosen%d-harmful%d trainset with %d base behaviors; chosen_indices=%s",
            len(args.chosen_indices),
            args.num_harmful_samples,
            len(trainset),
            args.chosen_indices,
        )
    return trainset


def main(args, logger, device):
    lora_dump_path = f"{args.save_dir}/{args.save_name}_lora-cfg.yaml" if is_main_process() else None
    lora_cfg = utils.load_yaml(args.lora_cfg_path, lora_dump_path)
    model, tokenizer = get_model(args.model_id, device, lora_cfg)
    model = maybe_wrap_ddp(model, device)
    trainset = get_trainset(args, logger)

    utilityset = None
    if args.utilityset is not None:
        utilityset = utils.get_dataset(args.utilityset)

    trainer_cfg = utils.load_yaml(args.trainer_cfg_path)
    trainer_cfg["name"] = "chosen_consistency_harmful_dpo"
    trainer_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    trainer_cfg["wandb_run_name"] = (
        f"qwen_dpo_260618_chosen_consistency_idx{indices_tag(args.chosen_indices)}_"
        f"h{args.num_harmful_samples}_beta{trainer_cfg['dpo_beta']}"
    )
    trainer_cfg["chosen_indices"] = list(args.chosen_indices)
    trainer_cfg["num_harmful_samples"] = args.num_harmful_samples
    trainer_cfg["listwise_reduce"] = args.listwise_reduce
    trainer_cfg["harmful_reduce"] = args.harmful_reduce
    trainer_cfg["listwise_temperature"] = args.listwise_temperature
    trainer_cfg["chosen_consistency_alpha"] = args.chosen_consistency_alpha
    trainer_cfg["harmful_consistency_alpha"] = args.harmful_consistency_alpha
    scale_batch_sizes_for_ddp(trainer_cfg, logger, args.gradient_accumulation_steps)

    if "datacollator_config" not in trainer_cfg.keys():
        trainer_cfg["datacollator_config"] = {}
    if args.datacollator is not None:
        trainer_cfg["datacollator_config"]["name"] = args.datacollator

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        trainer_cfg["attacker_config"] = atker_cfg

    if is_main_process():
        utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")

    trainer_cfg = AdvTrainerConfig(**trainer_cfg)
    trainer = AdvTrainerChosenConsistencyHarmfulDPO(
        trainer_config=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        trainset=trainset,
        utilityset=utilityset,
        logger=logger if is_main_process() else None,
    )
    if not is_main_process():
        trainer.wandb_run = None
    replace_loaders_for_ddp(trainer, trainset, utilityset, args.random_seed)

    model_state_dict, trainer_state_dict = None, None
    if args.model_resume_path is not None:
        if is_main_process():
            logger.info(f"resume model from {args.model_resume_path}")
        model_state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
    if args.trainer_resume_path is not None:
        if is_main_process():
            logger.info(f"resume trainer from {args.trainer_resume_path}")
        trainer_state_dict = utils.load_state_dict(args.trainer_resume_path, mode="pytorch")
    trainer.load_state_dict(model_state_dict, trainer_state_dict)
    del model_state_dict, trainer_state_dict

    trainer.train()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if is_main_process():
        _, trainer_state_dict = trainer.state_dict()
        model.save_pretrained(f"{args.save_dir}/{args.save_name}_fin-model")
        utils.save_state_dict(
            trainer_state_dict,
            path=f"{args.save_dir}/{args.save_name}_fin-trainer.pkl",
            mode="pytorch",
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = get_args()
    device = setup_distributed()
    logger = build_logger(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed + get_rank())
        torch.manual_seed(args.random_seed + get_rank())

    try:
        main(args, logger, device)
    except Exception as e:
        if is_main_process():
            logger.exception("Unexpected exception! %s", e)
        raise
    finally:
        cleanup_distributed()
