import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import utils
from adv_trainers import trainer_dpo2 as trainer_impl
from chosen4_fixed_data import HarmBenchChosen4Fixed02


@dataclass
class AdvTrainerConfig(trainer_impl.AdvTrainerConfig):
    margin_base: float = 0.0
    rejected_margin_alpha: float = 1.0
    rejected_margin_floor: float = 0.0
    margin_detach: bool = True
    wandb_run_name: str = None


class PairwiseDPOTrainerBase(trainer_impl.AdvTrainerDPO2):
    def _compute_pairwise_terms(self, inputs, advsfx_inputs):
        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        pi_chosen_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=inputs["benign_ids"],
            response_mask=inputs["benign_mask"],
        )
        pi_rejected_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=inputs["harmful_ids"],
            response_mask=inputs["harmful_mask"],
        )

        with torch.no_grad():
            ref_chosen_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=inputs["benign_ids"],
                response_mask=inputs["benign_mask"],
            )
            ref_rejected_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=inputs["harmful_ids"],
                response_mask=inputs["harmful_mask"],
            )

        delta_pi = pi_chosen_logp - pi_rejected_logp
        delta_ref = ref_chosen_logp - ref_rejected_logp
        reward_chosen = self.dpo_beta * (pi_chosen_logp - ref_chosen_logp)
        reward_rejected = self.dpo_beta * (pi_rejected_logp - ref_rejected_logp)

        return {
            "pi_chosen_logp": pi_chosen_logp,
            "pi_rejected_logp": pi_rejected_logp,
            "ref_chosen_logp": ref_chosen_logp,
            "ref_rejected_logp": ref_rejected_logp,
            "delta_pi": delta_pi,
            "delta_ref": delta_ref,
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
            "reward_margin": reward_chosen - reward_rejected,
        }

    def _backward_utility_loss(self, utility_inputs, scale):
        utility_loss = 0.0
        utility_alpha_loss = 0.0
        if utility_inputs is None:
            return utility_loss, utility_alpha_loss

        u_bs = len(utility_inputs["input_ids"])
        u_cnt = utility_inputs["input_mask"].sum()
        sp_bs = self.utility_split_bs
        if sp_bs is None:
            sp_bs = u_bs

        for be in range(0, u_bs, sp_bs):
            ed = min(u_bs, be + sp_bs)
            u_ids = utility_inputs["input_ids"][be:ed]
            u_mask = utility_inputs["input_mask"][be:ed]

            if self.utility_max_length is not None:
                max_len = self.utility_max_length
                if u_ids.shape[1] > max_len:
                    u_ids = u_ids[:, :max_len]
                    u_mask = u_mask[:, :max_len]

            u_logits = self.model(input_ids=u_ids).logits
            u_logps = u_logits.log_softmax(dim=-1)
            u_logps = torch.gather(
                u_logps[:, :-1],
                dim=-1,
                index=u_ids[:, 1:].unsqueeze(-1),
            ).squeeze(-1)
            u_mask = u_mask[:, 1:]

            u_loss = -(u_logps * u_mask).sum() / u_cnt
            (u_loss * self.utility_alpha * scale).backward()
            utility_loss += u_loss.item()
            utility_alpha_loss += (u_loss * self.utility_alpha).item()

        return utility_loss, utility_alpha_loss

    def _build_metric_dict(self, align_loss, align_alpha_loss, utility_loss, utility_alpha_loss, terms):
        delta_pi = terms["delta_pi"]
        delta_ref = terms["delta_ref"]
        reward_chosen = terms["reward_chosen"]
        reward_rejected = terms["reward_rejected"]
        reward_margin = terms["reward_margin"]

        return {
            "align_loss": align_loss.item(),
            "align_alpha_loss": align_alpha_loss.item(),
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss.item() + utility_alpha_loss,
            "dpo_margin": delta_pi.mean().item(),
            "dpo_acc": (delta_pi > delta_ref).to(torch.float32).mean().item(),
            "reward_chosen": reward_chosen.mean().item(),
            "reward_rejected": reward_rejected.mean().item(),
            "reward_margin": reward_margin.mean().item(),
            "reward_winner_rate": (reward_chosen > reward_rejected).to(torch.float32).mean().item(),
            "pi_logp_chosen": terms["pi_chosen_logp"].mean().item(),
            "pi_logp_rejected": terms["pi_rejected_logp"].mean().item(),
            "ref_logp_chosen": terms["ref_chosen_logp"].mean().item(),
            "ref_logp_rejected": terms["ref_rejected_logp"].mean().item(),
        }


class AdvTrainerAdaptiveMarginDPO(PairwiseDPOTrainerBase):
    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.margin_base = trainer_config.margin_base
        self.rejected_margin_alpha = trainer_config.rejected_margin_alpha
        self.rejected_margin_floor = trainer_config.rejected_margin_floor
        self.margin_detach = trainer_config.margin_detach
        self.wandb_run_name = trainer_config.wandb_run_name
        super().__init__(trainer_config=trainer_config, **kwargs)

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or f"adaptive-margin-dpo-beta-{self.dpo_beta}"
            self.wandb_run.config.update({
                "loss_type": "adaptive_margin_dpo",
                "margin_base": self.margin_base,
                "rejected_margin_alpha": self.rejected_margin_alpha,
                "rejected_margin_floor": self.rejected_margin_floor,
                "margin_detach": self.margin_detach,
            })

        if self.logger is not None:
            self.logger.info(
                "Adaptive-Margin DPO enabled: margin_base=%.4f, rejected_margin_alpha=%.4f, "
                "rejected_margin_floor=%.4f, margin_detach=%s, dpo_beta=%.4f",
                self.margin_base,
                self.rejected_margin_alpha,
                self.rejected_margin_floor,
                self.margin_detach,
                self.dpo_beta,
            )

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps
        terms = self._compute_pairwise_terms(inputs=inputs, advsfx_inputs=advsfx_inputs)

        margin_source = terms["reward_rejected"]
        if self.margin_detach:
            margin_source = margin_source.detach()
        adaptive_margin = self.margin_base + self.rejected_margin_alpha * F.relu(
            margin_source - self.rejected_margin_floor
        )

        logits = self.dpo_beta * (terms["delta_pi"] - terms["delta_ref"]) - adaptive_margin
        per_sample_loss = -F.logsigmoid(logits)
        align_loss = per_sample_loss.mean()
        align_alpha_loss = align_loss * self.adv_alpha
        (align_alpha_loss * scale).backward()

        utility_loss, utility_alpha_loss = self._backward_utility_loss(utility_inputs, scale)
        metrics = self._build_metric_dict(
            align_loss=align_loss,
            align_alpha_loss=align_alpha_loss,
            utility_loss=utility_loss,
            utility_alpha_loss=utility_alpha_loss,
            terms=terms,
        )
        metrics.update({
            "adaptive_margin": adaptive_margin.mean().item(),
            "adaptive_margin_max": adaptive_margin.max().item(),
        })
        return metrics


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)
    utils.add_train_args(parser)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def get_pair_trainset(args, logger):
    if args.dataset == "harmbench-chosen4":
        trainset = HarmBenchChosen4Fixed02()
        logger.info(
            "Using fixed chosen4 indices %s; expanded trainset from %d base behaviors to %d samples",
            trainset.fixed_chosen_indices,
            len(trainset.prompt_ds),
            len(trainset),
        )
        return trainset

    return utils.get_dataset(args.dataset)


def run_training(
    args,
    logger,
    trainer_cls,
    config_cls,
    trainer_name,
    wandb_run_name,
    extra_trainer_cfg=None,
    trainset_fn=get_pair_trainset,
    double_chosen4_steps=True,
):
    default_device = "cuda:0"

    lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    trainset = trainset_fn(args, logger)

    utilityset = None
    if args.utilityset is not None:
        utilityset = utils.get_dataset(args.utilityset)

    trainer_cfg = utils.load_yaml(args.trainer_cfg_path)
    trainer_cfg["name"] = trainer_name
    trainer_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    trainer_cfg["wandb_run_name"] = wandb_run_name.format(beta=trainer_cfg["dpo_beta"])
    if extra_trainer_cfg is not None:
        trainer_cfg.update(extra_trainer_cfg)

    if double_chosen4_steps and args.dataset == "harmbench-chosen4":
        base_train_steps = trainer_cfg["train_steps"]
        trainer_cfg["train_steps"] = int(base_train_steps * 2)
        logger.info(
            "Using fixed chosen4 index subset [0, 2]; overriding train_steps from %d to %d",
            base_train_steps,
            trainer_cfg["train_steps"],
        )

    if "datacollator_config" not in trainer_cfg.keys():
        trainer_cfg["datacollator_config"] = {}
    if args.datacollator is not None:
        trainer_cfg["datacollator_config"]["name"] = args.datacollator

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        trainer_cfg["attacker_config"] = atker_cfg

    utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")
    trainer_cfg = config_cls(**trainer_cfg)

    trainer = trainer_cls(
        trainer_config=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
        trainset=trainset,
        utilityset=utilityset,
        logger=logger,
    )

    model_state_dict, trainer_state_dict = None, None
    if args.model_resume_path is not None:
        logger.info(f"resume model from {args.model_resume_path}")
        model_state_dict = utils.load_state_dict(args.model_resume_path, mode="safetensors")
    if args.trainer_resume_path is not None:
        logger.info(f"resume trainer from {args.trainer_resume_path}")
        trainer_state_dict = utils.load_state_dict(args.trainer_resume_path, mode="pytorch")
    trainer.load_state_dict(model_state_dict, trainer_state_dict)
    del model_state_dict, trainer_state_dict

    trainer.train()

    _, trainer_state_dict = trainer.state_dict()
    model.save_pretrained(f"{args.save_dir}/{args.save_name}_fin-model")
    utils.save_state_dict(
        trainer_state_dict,
        path=f"{args.save_dir}/{args.save_name}_fin-trainer.pkl",
        mode="pytorch",
    )


def main(args, logger):
    run_training(
        args=args,
        logger=logger,
        trainer_cls=AdvTrainerAdaptiveMarginDPO,
        config_cls=AdvTrainerConfig,
        trainer_name="adaptive_margin_dpo",
        wandb_run_name="adaptive-margin-dpo-beta-{beta}",
    )


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception("Unexpected exception! %s", e)
