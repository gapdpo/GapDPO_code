from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import utils
from dpo_common import get_args, run_training
from dpo_listwise import (
    AdvTrainerConfig as ListwiseAdvTrainerConfig,
    AdvTrainerListwiseChosen4DPO,
    get_listwise_trainset,
)


@dataclass
class AdvTrainerConfig(ListwiseAdvTrainerConfig):
    listwise_reduce: str = "mean"
    chosen_consistency_alpha: float = 0.1


class AdvTrainerChosenConsistencyDPO(AdvTrainerListwiseChosen4DPO):
    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.chosen_consistency_alpha = trainer_config.chosen_consistency_alpha
        super().__init__(trainer_config=trainer_config, **kwargs)

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or f"chosen-consistency-dpo-beta-{self.dpo_beta}"
            self.wandb_run.config.update({
                "loss_type": "chosen_consistency_dpo",
                "chosen_consistency_alpha": self.chosen_consistency_alpha,
            }, allow_val_change=True)

        if self.logger is not None:
            self.logger.info(
                "Chosen-Consistency DPO enabled: chosen_consistency_alpha=%.4f, reduce=%s, dpo_beta=%.4f",
                self.chosen_consistency_alpha,
                self.listwise_reduce,
                self.dpo_beta,
            )

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps
        terms = self._compute_listwise_terms(inputs=inputs, advsfx_inputs=advsfx_inputs)

        dpo_loss = -F.logsigmoid(terms["reward_margin"]).mean()
        consistency_loss = terms["reward_chosen"].var(dim=1, unbiased=False).mean()
        align_loss = dpo_loss + (self.chosen_consistency_alpha * consistency_loss)
        align_alpha_loss = align_loss * self.adv_alpha
        (align_alpha_loss * scale).backward()

        utility_loss, utility_alpha_loss = self._backward_utility_loss(utility_inputs, scale)
        reward_chosen = terms["reward_chosen"]
        reward_rejected = terms["reward_rejected"]
        aggregate_reward_chosen = terms["aggregate_reward_chosen"]

        return {
            "align_loss": align_loss.item(),
            "align_alpha_loss": align_alpha_loss.item(),
            "dpo_loss": dpo_loss.item(),
            "chosen_consistency_loss": consistency_loss.item(),
            "chosen_consistency_alpha_loss": (self.chosen_consistency_alpha * consistency_loss).item(),
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


def main(args, logger):
    run_training(
        args=args,
        logger=logger,
        trainer_cls=AdvTrainerChosenConsistencyDPO,
        config_cls=AdvTrainerConfig,
        trainer_name="chosen_consistency_dpo",
        wandb_run_name="chosen-consistency-dpo-beta-{beta}",
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
