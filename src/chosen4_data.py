import json
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


class HarmBenchChosen4Safety(HarmBench):
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
            if len(self.target_ds[behavior_id]) < 1:
                raise ValueError(f"behavior_id={behavior_id} has no harmful targets")

    def __getitem__(self, idx):
        sample = self.prompt_ds[idx]
        behavior_id = sample["BehaviorID"]
        return {
            "prompt": sample["Behavior"],
            "chosen_list": self.chosen_ds[behavior_id],
            "harmful_pool": self.target_ds[behavior_id],
        }


class SafetyEnvelopeDatacollator:
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

    def __init__(self, tokenizer, name: str, chosen_indices, num_harmful_samples: int):
        if name not in self.PREFIXES:
            raise ValueError(f"Unsupported datacollator name={name}")
        if num_harmful_samples < 1:
            raise ValueError(f"num_harmful_samples must be positive, got {num_harmful_samples}")
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
        self.chosen_indices = tuple(chosen_indices)
        self.num_harmful_samples = num_harmful_samples
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

    def _sample_harmful_list(self, harmful_pool):
        if len(harmful_pool) < 1:
            raise ValueError("harmful_pool must contain at least one target")
        replace = len(harmful_pool) < self.num_harmful_samples
        indices = np.random.choice(len(harmful_pool), size=self.num_harmful_samples, replace=replace)
        return [harmful_pool[idx] for idx in indices]

    def __call__(self, features):
        prompt = [self._encode_prompt(feat["prompt"]) for feat in features]
        prompt_ids, prompt_mask = self._pad_left(prompt)

        chosen_encoded = []
        harmful_encoded = []
        for feat in features:
            chosen_list = feat["chosen_list"]
            if max(self.chosen_indices) >= len(chosen_list):
                raise ValueError(
                    f"chosen_list has length {len(chosen_list)}; cannot use indices {self.chosen_indices}"
                )
            chosen_encoded.extend(self._encode_response(chosen_list[idx]) for idx in self.chosen_indices)

            sampled_harmful = self._sample_harmful_list(feat["harmful_pool"])
            harmful_encoded.extend(self._encode_response(text) for text in sampled_harmful)

        chosen_ids, chosen_mask = self._pad_right(chosen_encoded)
        harmful_ids, harmful_mask = self._pad_right(harmful_encoded)

        bs = len(features)
        num_chosen = len(self.chosen_indices)
        chosen_ids = chosen_ids.reshape(bs, num_chosen, -1)
        chosen_mask = chosen_mask.reshape(bs, num_chosen, -1)
        harmful_ids = harmful_ids.reshape(bs, self.num_harmful_samples, -1)
        harmful_mask = harmful_mask.reshape(bs, self.num_harmful_samples, -1)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "harmful_ids": harmful_ids,
            "harmful_mask": harmful_mask,
        }


@dataclass
class AdvTrainerConfig(trainer_impl.AdvTrainerConfig):
    chosen_indices: tuple = field(default_factory=lambda: (0, 1, 2, 3))
    num_harmful_samples: int = 4
    chosen_std_alpha: float = 0.5
    harmful_std_alpha: float = 1.0
    safety_margin: float = 0.0
    harmful_reward_ceiling: float = 0.0
    harmful_overflow_alpha: float = 0.05
    wandb_run_name: str = None


class AdvTrainerSafetyEnvelopeDPO(PairwiseDPOTrainerBase):
    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.chosen_indices = tuple(trainer_config.chosen_indices)
        self.num_harmful_samples = trainer_config.num_harmful_samples
        self.chosen_std_alpha = trainer_config.chosen_std_alpha
        self.harmful_std_alpha = trainer_config.harmful_std_alpha
        self.safety_margin = trainer_config.safety_margin
        self.harmful_reward_ceiling = trainer_config.harmful_reward_ceiling
        self.harmful_overflow_alpha = trainer_config.harmful_overflow_alpha
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

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or f"safety-envelope-dpo-beta-{self.dpo_beta}"
            self.wandb_run.config.update({
                "loss_type": "safety_envelope_dpo",
                "chosen_indices": self.chosen_indices,
                "num_harmful_samples": self.num_harmful_samples,
                "chosen_std_alpha": self.chosen_std_alpha,
                "harmful_std_alpha": self.harmful_std_alpha,
                "safety_margin": self.safety_margin,
                "harmful_reward_ceiling": self.harmful_reward_ceiling,
                "harmful_overflow_alpha": self.harmful_overflow_alpha,
            })

        if self.logger is not None:
            self.logger.info(
                "Safety-Envelope DPO enabled: chosen_indices=%s, num_harmful=%d, "
                "chosen_std_alpha=%.4f, harmful_std_alpha=%.4f, safety_margin=%.4f, "
                "harmful_reward_ceiling=%.4f, harmful_overflow_alpha=%.4f, dpo_beta=%.4f",
                self.chosen_indices,
                self.num_harmful_samples,
                self.chosen_std_alpha,
                self.harmful_std_alpha,
                self.safety_margin,
                self.harmful_reward_ceiling,
                self.harmful_overflow_alpha,
                self.dpo_beta,
            )

    def _compute_safety_terms(self, inputs, advsfx_inputs):
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

        chosen_mean = reward_chosen.mean(dim=1)
        harmful_mean = reward_harmful.mean(dim=1)
        chosen_std = reward_chosen.std(dim=1, unbiased=False)
        harmful_std = reward_harmful.std(dim=1, unbiased=False)
        safe_anchor = chosen_mean - (self.chosen_std_alpha * chosen_std)
        harmful_anchor = harmful_mean + (self.harmful_std_alpha * harmful_std)
        safety_gap = safe_anchor - harmful_anchor

        return {
            "pi_chosen_logp": pi_chosen_logp,
            "pi_harmful_logp": pi_harmful_logp,
            "ref_chosen_logp": ref_chosen_logp,
            "ref_harmful_logp": ref_harmful_logp,
            "reward_chosen": reward_chosen,
            "reward_harmful": reward_harmful,
            "chosen_mean": chosen_mean,
            "harmful_mean": harmful_mean,
            "chosen_std": chosen_std,
            "harmful_std": harmful_std,
            "safe_anchor": safe_anchor,
            "harmful_anchor": harmful_anchor,
            "safety_gap": safety_gap,
        }

    def _build_attack_target_inputs(self, inputs):
        harmful_ids = inputs["harmful_ids"]
        harmful_mask = inputs["harmful_mask"]

        if harmful_ids.dim() == 2:
            return harmful_ids, harmful_mask
        if harmful_ids.dim() != 3:
            raise ValueError(f"Expected harmful_ids to be 2D or 3D, got shape={tuple(harmful_ids.shape)}")

        # GCG builds one suffix per prompt, while safety-envelope DPO scores a pool
        # of harmful responses. Use one sampled harmful target for suffix search and
        # keep the full pool for the envelope loss below.
        return harmful_ids[:, 0, :].contiguous(), harmful_mask[:, 0, :].contiguous()

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps
        terms = self._compute_safety_terms(inputs=inputs, advsfx_inputs=advsfx_inputs)

        envelope_loss = F.softplus(self.safety_margin - terms["safety_gap"]).mean()
        harmful_overflow = F.relu(terms["reward_harmful"] - self.harmful_reward_ceiling).pow(2).mean()
        align_loss = envelope_loss + (self.harmful_overflow_alpha * harmful_overflow)
        align_alpha_loss = align_loss * self.adv_alpha
        (align_alpha_loss * scale).backward()

        utility_loss, utility_alpha_loss = self._backward_utility_loss(utility_inputs, scale)
        safety_acc = (terms["safe_anchor"] > terms["harmful_anchor"]).to(torch.float32).mean().item()
        pairwise_win_rate = (
            terms["reward_chosen"].unsqueeze(2) > terms["reward_harmful"].unsqueeze(1)
        ).to(torch.float32).mean().item()

        return {
            "align_loss": align_loss.item(),
            "align_alpha_loss": align_alpha_loss.item(),
            "envelope_loss": envelope_loss.item(),
            "harmful_overflow_loss": harmful_overflow.item(),
            "harmful_overflow_alpha_loss": (self.harmful_overflow_alpha * harmful_overflow).item(),
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss.item() + utility_alpha_loss,
            "dpo_margin": terms["safety_gap"].mean().item(),
            "dpo_acc": pairwise_win_rate,
            "safety_gap": terms["safety_gap"].mean().item(),
            "safety_acc": safety_acc,
            "pairwise_win_rate": pairwise_win_rate,
            "safe_anchor": terms["safe_anchor"].mean().item(),
            "harmful_anchor": terms["harmful_anchor"].mean().item(),
            "reward_chosen": terms["reward_chosen"].mean().item(),
            "reward_harmful": terms["reward_harmful"].mean().item(),
            "reward_margin": terms["safety_gap"].mean().item(),
            "reward_chosen_std": terms["chosen_std"].mean().item(),
            "reward_harmful_std": terms["harmful_std"].mean().item(),
            "reward_harmful_max": terms["reward_harmful"].max().item(),
            "pi_logp_chosen": terms["pi_chosen_logp"].mean().item(),
            "pi_logp_harmful": terms["pi_harmful_logp"].mean().item(),
            "ref_logp_chosen": terms["ref_chosen_logp"].mean().item(),
            "ref_logp_harmful": terms["ref_harmful_logp"].mean().item(),
        }


def get_safety_trainset(args, logger):
    if args.dataset != "harmbench-chosen4":
        raise ValueError("Safety-Envelope DPO requires --dataset harmbench-chosen4")
    trainset = HarmBenchChosen4Safety()
    logger.info(
        "Using safety-envelope chosen4 trainset with %d base behaviors and chosen/harmful response pools",
        len(trainset),
    )
    return trainset


def main(args, logger):
    run_training(
        args=args,
        logger=logger,
        trainer_cls=AdvTrainerSafetyEnvelopeDPO,
        config_cls=AdvTrainerConfig,
        trainer_name="safety_envelope_dpo",
        wandb_run_name="safety-envelope-dpo-beta-{beta}",
        trainset_fn=get_safety_trainset,
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
