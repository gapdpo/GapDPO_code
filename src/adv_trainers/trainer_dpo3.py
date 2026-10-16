from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import trainer_dpo2 as trainer_dpo2_impl


AdvTrainerBase = trainer_dpo2_impl.AdvTrainerBase
AdvTrainerSFT = trainer_dpo2_impl.AdvTrainerSFT


@dataclass
class AdvTrainerConfig(trainer_dpo2_impl.AdvTrainerConfig):
    chosen_sft_alpha: float = 0.1
    wandb_run_name: str = None


class AdvTrainerDPO3(trainer_dpo2_impl.AdvTrainerDPO2):

    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.chosen_sft_alpha = trainer_config.chosen_sft_alpha
        self.wandb_run_name = trainer_config.wandb_run_name
        super().__init__(trainer_config=trainer_config, **kwargs)

        if self.wandb_run is not None:
            if self.wandb_run_name is not None:
                self.wandb_run.name = self.wandb_run_name
            else:
                self.wandb_run.name = f"dpo3-beta-{trainer_config.dpo_beta}"
            self.wandb_run.config.update({
                "chosen_sft_alpha": self.chosen_sft_alpha,
            })

        if self.logger is not None:
            self.logger.info(
                "Hybrid DPO enabled: chosen_sft_alpha=%.4f, adv_alpha=%.4f, dpo_beta=%.4f",
                self.chosen_sft_alpha,
                self.adv_alpha,
                self.dpo_beta,
            )

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps

        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        chosen_ids = inputs["benign_ids"]
        chosen_mask = inputs["benign_mask"]
        rejected_ids = inputs["harmful_ids"]
        rejected_mask = inputs["harmful_mask"]

        pi_chosen_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=chosen_ids,
            response_mask=chosen_mask,
        )
        pi_rejected_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=rejected_ids,
            response_mask=rejected_mask,
        )

        with torch.no_grad():
            ref_chosen_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=chosen_ids,
                response_mask=chosen_mask,
            )
            ref_rejected_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=rejected_ids,
                response_mask=rejected_mask,
            )

        delta_pi = pi_chosen_logp - pi_rejected_logp
        delta_ref = ref_chosen_logp - ref_rejected_logp
        reward_chosen = self.dpo_beta * (pi_chosen_logp - ref_chosen_logp)
        reward_rejected = self.dpo_beta * (pi_rejected_logp - ref_rejected_logp)
        reward_margin = reward_chosen - reward_rejected

        dpo_loss = -F.logsigmoid(self.dpo_beta * (delta_pi - delta_ref)).mean()
        chosen_sft_loss = -pi_chosen_logp.mean()
        hybrid_align_loss = dpo_loss + (self.chosen_sft_alpha * chosen_sft_loss)
        (hybrid_align_loss * self.adv_alpha * scale).backward()

        dpo_margin = delta_pi.mean().item()
        dpo_acc = (delta_pi > delta_ref).to(torch.float32).mean().item()
        reward_winner_rate = (reward_chosen > reward_rejected).to(torch.float32).mean().item()
        align_alpha_loss = (hybrid_align_loss * self.adv_alpha).item()

        utility_loss = 0.0
        utility_alpha_loss = 0.0
        if utility_inputs is not None:
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
                    max_L = self.utility_max_length
                    if u_ids.shape[1] > max_L:
                        u_ids = u_ids[:, :max_L]
                        u_mask = u_mask[:, :max_L]

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

        return {
            "align_loss": hybrid_align_loss.item(),
            "align_alpha_loss": align_alpha_loss,
            "dpo_loss": dpo_loss.item(),
            "chosen_sft_loss": chosen_sft_loss.item(),
            "chosen_sft_alpha_loss": (self.chosen_sft_alpha * chosen_sft_loss).item(),
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss + utility_alpha_loss,
            "dpo_margin": dpo_margin,
            "dpo_acc": dpo_acc,
            "reward_chosen": reward_chosen.mean().item(),
            "reward_rejected": reward_rejected.mean().item(),
            "reward_margin": reward_margin.mean().item(),
            "reward_winner_rate": reward_winner_rate,
            "pi_logp_chosen": pi_chosen_logp.mean().item(),
            "pi_logp_rejected": pi_rejected_logp.mean().item(),
            "ref_logp_chosen": ref_chosen_logp.mean().item(),
            "ref_logp_rejected": ref_rejected_logp.mean().item(),
        }


__adv_trainer_zoo__ = {
    "sft": AdvTrainerSFT,
    "dpo3": AdvTrainerDPO3,
}


def build_trainer(trainer_config: AdvTrainerConfig, **kwargs):
    return __adv_trainer_zoo__[trainer_config.name](
        trainer_config=trainer_config,
        **kwargs,
    )
