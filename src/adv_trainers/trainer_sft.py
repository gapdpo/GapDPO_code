from dataclasses import dataclass

import torch

from . import trainer_dpo2 as trainer_dpo2_impl


AdvTrainerBase = trainer_dpo2_impl.AdvTrainerBase


@dataclass
class AdvTrainerConfig(trainer_dpo2_impl.AdvTrainerConfig):
    wandb_run_name: str = None


class AdvTrainerSFT260504(trainer_dpo2_impl.AdvTrainerDPO2):

    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.wandb_run_name = trainer_config.wandb_run_name
        super().__init__(trainer_config=trainer_config, **kwargs)

        if self.wandb_run is not None:
            self.wandb_run.name = self.wandb_run_name or "sft_260504"

        if self.logger is not None:
            self.logger.info(
                "Chosen-only SFT enabled: adv_alpha=%.4f, utility_alpha=%.4f",
                self.adv_alpha,
                self.utility_alpha,
            )

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps

        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        chosen_ids = inputs["benign_ids"]
        chosen_mask = inputs["benign_mask"]

        pi_chosen_logp = self._compute_response_logp(
            model=self.model,
            context_inputs=context_inputs,
            response_ids=chosen_ids,
            response_mask=chosen_mask,
        )

        sft_loss = -pi_chosen_logp.mean()
        (sft_loss * self.adv_alpha * scale).backward()

        align_alpha_loss = (sft_loss * self.adv_alpha).item()
        sft_loss_val = sft_loss.item()

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
            "align_loss": sft_loss_val,
            "align_alpha_loss": align_alpha_loss,
            "sft_loss": sft_loss_val,
            "utility_loss": utility_loss,
            "utility_alpha_loss": utility_alpha_loss,
            "total_loss": align_alpha_loss + utility_alpha_loss,
            "pi_logp_chosen": pi_chosen_logp.mean().item(),
        }


__adv_trainer_zoo__ = {
    "sft": AdvTrainerSFT260504,
}


def build_trainer(trainer_config: AdvTrainerConfig, **kwargs):
    return __adv_trainer_zoo__[trainer_config.name](
        trainer_config=trainer_config,
        **kwargs,
    )
