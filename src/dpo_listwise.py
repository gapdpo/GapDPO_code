import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

import utils
from adv_trainers import trainer_dpo2 as trainer_impl
from adv_trainers.data import Loader
from dpo_common import (
    PairwiseDPOTrainerBase,
    get_args,
    run_training,
)
from utils.datasets import HarmBench, resolve_data_root


class HarmBenchChosen4List(HarmBench):
    def __init__(self, root: str = "../data"):
        root = resolve_data_root(root)
        super().__init__(root=root, split="train")
        chosen_path = "harmbench/optimizer_targets/extra_targets/adv_training_targets_chosen4.json"
        self.chosen_path = os.path.join(self.root, chosen_path)
        with open(self.chosen_path, "rb") as f:
            self.chosen_ds = json.load(f)

        for behavior_id, chosen_list in self.chosen_ds.items():
            if len(chosen_list) < 2:
                raise ValueError(f"behavior_id={behavior_id} has fewer than 2 chosen responses")

    def __getitem__(self, idx):
        sample = self.prompt_ds[idx]
        behavior_id = sample["BehaviorID"]
        behavior_list = self.target_ds[behavior_id]
        return {
            "prompt": sample["Behavior"],
            "target": behavior_list[np.random.randint(len(behavior_list))],
            "chosen_list": self.chosen_ds[behavior_id],
        }


class ListwiseChosen4Datacollator:
    PREFIXES = {
        "vicuna-chat": ("USER:", "\nASSISTANT:"),
        "llama2-chat": ("[INST]", "[/INST]"),
        "llama3-chat": (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        "qwen2-chat": (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        ),
    }

    def __init__(self, tokenizer, name: str, chosen_indices):
        if name not in self.PREFIXES:
            raise ValueError(f"Unsupported datacollator name={name}")
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        self.chosen_indices = tuple(chosen_indices)
        prompt_prefix, target_prefix = self.PREFIXES[name]
        self.prompt_prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=True, return_tensors="pt")[0]
        self.target_prefix_ids = tokenizer.encode(target_prefix, add_special_tokens=False, return_tensors="pt")[0]
        self.prompt_prefix_mask = torch.ones(len(self.prompt_prefix_ids), dtype=torch.int64)
        self.target_prefix_mask = torch.zeros(len(self.target_prefix_ids), dtype=torch.int64)

    def _encode_prompt(self, text):
        ids = self.tokenizer.encode(text.lstrip().rstrip(), add_special_tokens=False, return_tensors="pt")[0]
        ids = torch.cat([self.prompt_prefix_ids, ids])
        mask = torch.cat([self.prompt_prefix_mask, torch.ones(len(ids) - len(self.prompt_prefix_ids), dtype=torch.int64)])
        return ids, mask

    def _encode_response(self, text):
        ids = self.tokenizer.encode(text.lstrip().rstrip(), add_special_tokens=False, return_tensors="pt")[0]
        ids = torch.cat([self.target_prefix_ids, ids])
        mask = torch.cat([self.target_prefix_mask, torch.ones(len(ids) - len(self.target_prefix_ids), dtype=torch.int64)])
        return ids, mask

    def _pad_left(self, encoded):
        bs = len(encoded)
        max_len = max(len(ids) for ids, _ in encoded)
        ids_out = torch.full((bs, max_len), fill_value=self.pad_id, dtype=torch.int64)
        mask_out = torch.zeros_like(ids_out)
        for i, (ids, mask) in enumerate(encoded):
            ids_out[i, -len(ids):] = ids
            mask_out[i, -len(ids):] = mask
        return ids_out, mask_out

    def _pad_right(self, encoded):
        bs = len(encoded)
        max_len = max(len(ids) for ids, _ in encoded)
        ids_out = torch.full((bs, max_len), fill_value=self.pad_id, dtype=torch.int64)
        mask_out = torch.zeros_like(ids_out)
        for i, (ids, mask) in enumerate(encoded):
            ids_out[i, :len(ids)] = ids
            mask_out[i, :len(ids)] = mask
        return ids_out, mask_out

    def __call__(self, features):
        prompt = [self._encode_prompt(feat["prompt"]) for feat in features]
        harmful = [self._encode_response(feat["target"]) for feat in features]
        prompt_ids, prompt_mask = self._pad_left(prompt)
        harmful_ids, harmful_mask = self._pad_right(harmful)

        chosen_encoded = []
        for feat in features:
            chosen_list = feat["chosen_list"]
            if max(self.chosen_indices) >= len(chosen_list):
                raise ValueError(
                    f"chosen_list has length {len(chosen_list)}; cannot use indices {self.chosen_indices}"
                )
            chosen_encoded.extend(self._encode_response(chosen_list[idx]) for idx in self.chosen_indices)

        chosen_ids, chosen_mask = self._pad_right(chosen_encoded)
        bs = len(features)
        num_chosen = len(self.chosen_indices)
        chosen_ids = chosen_ids.view(bs, num_chosen, -1)
        chosen_mask = chosen_mask.view(bs, num_chosen, -1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "harmful_ids": harmful_ids,
            "harmful_mask": harmful_mask,
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
        }


@dataclass
class AdvTrainerConfig(trainer_impl.AdvTrainerConfig):
    chosen_indices: tuple = field(default_factory=lambda: (0, 1, 2, 3))
    listwise_reduce: str = "min"
    listwise_temperature: float = 1.0
    wandb_run_name: str = None


class AdvTrainerListwiseChosen4DPO(PairwiseDPOTrainerBase):
    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.chosen_indices = tuple(trainer_config.chosen_indices)
        self.listwise_reduce = trainer_config.listwise_reduce
        self.listwise_temperature = trainer_config.listwise_temperature
        self.wandb_run_name = trainer_config.wandb_run_name
        super().__init__(trainer_config=trainer_config, **kwargs)

        self.train_datacollator = ListwiseChosen4Datacollator(
            tokenizer=self.tokenizer,
            name=trainer_config.datacollator_config.name,
            chosen_indices=self.chosen_indices,
        )
        self.trainloader = Loader(
            dataset=self.trainset,
            batch_size=trainer_config.batch_size,
            train=True,
            collate_fn=self.train_datacollator,
            num_workers=0,
        )

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or f"listwise-chosen4-dpo-beta-{self.dpo_beta}"
            self.wandb_run.config.update({
                "loss_type": "listwise_chosen4_dpo",
                "chosen_indices": self.chosen_indices,
                "listwise_reduce": self.listwise_reduce,
                "listwise_temperature": self.listwise_temperature,
            })

        if self.logger is not None:
            self.logger.info(
                "Listwise Chosen4 DPO enabled: chosen_indices=%s, reduce=%s, temperature=%.4f, dpo_beta=%.4f",
                self.chosen_indices,
                self.listwise_reduce,
                self.listwise_temperature,
                self.dpo_beta,
            )

    def _aggregate_chosen_reward(self, reward_chosen):
        if self.listwise_reduce == "min":
            return reward_chosen.min(dim=1).values
        if self.listwise_reduce == "mean":
            return reward_chosen.mean(dim=1)
        if self.listwise_reduce == "logsumexp":
            temp = self.listwise_temperature
            if temp <= 0:
                raise ValueError(f"listwise_temperature must be positive, got {temp}")
            return temp * (torch.logsumexp(reward_chosen / temp, dim=1) - math.log(reward_chosen.shape[1]))
        raise ValueError(f"Unsupported listwise_reduce={self.listwise_reduce}")

    def _compute_listwise_terms(self, inputs, advsfx_inputs):
        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        chosen_ids = inputs["chosen_ids"]
        chosen_mask = inputs["chosen_mask"]
        bs, num_chosen, chosen_len = chosen_ids.shape

        chosen_context = {
            "context_ids": context_inputs["context_ids"].repeat_interleave(num_chosen, dim=0),
            "context_mask": context_inputs["context_mask"].repeat_interleave(num_chosen, dim=0),
        }
        chosen_ids_flat = chosen_ids.view(bs * num_chosen, chosen_len)
        chosen_mask_flat = chosen_mask.view(bs * num_chosen, chosen_len)

        pi_chosen_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=chosen_context,
            response_ids=chosen_ids_flat,
            response_mask=chosen_mask_flat,
        ).view(bs, num_chosen)
        pi_rejected_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=inputs["harmful_ids"],
            response_mask=inputs["harmful_mask"],
        )

        with torch.no_grad():
            ref_chosen_logp = self._compute_reference_response_logp(
                context_inputs=chosen_context,
                response_ids=chosen_ids_flat,
                response_mask=chosen_mask_flat,
            ).view(bs, num_chosen)
            ref_rejected_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=inputs["harmful_ids"],
                response_mask=inputs["harmful_mask"],
            )

        reward_chosen = self.dpo_beta * (pi_chosen_logp - ref_chosen_logp)
        reward_rejected = self.dpo_beta * (pi_rejected_logp - ref_rejected_logp)
        aggregate_reward_chosen = self._aggregate_chosen_reward(reward_chosen)

        return {
            "pi_chosen_logp": pi_chosen_logp,
            "pi_rejected_logp": pi_rejected_logp,
            "ref_chosen_logp": ref_chosen_logp,
            "ref_rejected_logp": ref_rejected_logp,
            "reward_chosen": reward_chosen,
            "reward_rejected": reward_rejected,
            "aggregate_reward_chosen": aggregate_reward_chosen,
            "reward_margin": aggregate_reward_chosen - reward_rejected,
        }

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps
        terms = self._compute_listwise_terms(inputs=inputs, advsfx_inputs=advsfx_inputs)

        per_sample_loss = -F.logsigmoid(terms["reward_margin"])
        align_loss = per_sample_loss.mean()
        align_alpha_loss = align_loss * self.adv_alpha
        (align_alpha_loss * scale).backward()

        utility_loss, utility_alpha_loss = self._backward_utility_loss(utility_inputs, scale)
        reward_chosen = terms["reward_chosen"]
        reward_rejected = terms["reward_rejected"]
        aggregate_reward_chosen = terms["aggregate_reward_chosen"]

        return {
            "align_loss": align_loss.item(),
            "align_alpha_loss": align_alpha_loss.item(),
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss.item() + utility_alpha_loss,
            "listwise_dpo_acc": (aggregate_reward_chosen > reward_rejected).to(torch.float32).mean().item(),
            "reward_chosen": reward_chosen.mean().item(),
            "reward_chosen_min": reward_chosen.min(dim=1).values.mean().item(),
            "reward_chosen_max": reward_chosen.max(dim=1).values.mean().item(),
            "reward_rejected": reward_rejected.mean().item(),
            "reward_margin": terms["reward_margin"].mean().item(),
            "pi_logp_chosen": terms["pi_chosen_logp"].mean().item(),
            "pi_logp_rejected": terms["pi_rejected_logp"].mean().item(),
            "ref_logp_chosen": terms["ref_chosen_logp"].mean().item(),
            "ref_logp_rejected": terms["ref_rejected_logp"].mean().item(),
        }


def get_listwise_trainset(args, logger):
    if args.dataset != "harmbench-chosen4":
        raise ValueError("Listwise Chosen4 DPO requires --dataset harmbench-chosen4")
    trainset = HarmBenchChosen4List()
    logger.info(
        "Using listwise chosen4 trainset with %d base behaviors and all chosen candidates per sample",
        len(trainset),
    )
    return trainset


def main(args, logger):
    run_training(
        args=args,
        logger=logger,
        trainer_cls=AdvTrainerListwiseChosen4DPO,
        config_cls=AdvTrainerConfig,
        trainer_name="listwise_chosen4_dpo",
        wandb_run_name="listwise-chosen4-dpo-beta-{beta}",
        trainset_fn=get_listwise_trainset,
        double_chosen4_steps=False,
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
