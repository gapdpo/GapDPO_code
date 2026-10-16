from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import trainer_dpo1 as trainer_dpo1_impl

AdvTrainerBase = trainer_dpo1_impl.AdvTrainerBase
AdvTrainerSFT = trainer_dpo1_impl.AdvTrainerSFT


@dataclass
class AdvTrainerConfig(trainer_dpo1_impl.AdvTrainerConfig):
    gradient_accumulation_steps: int = 1


class AdvTrainerDPO2(AdvTrainerSFT):

    def __init__(self, trainer_config: AdvTrainerConfig, **kwargs):
        self.gradient_accumulation_steps = trainer_config.gradient_accumulation_steps
        effective_batch_size = trainer_config.batch_size
        effective_utility_batch_size = trainer_config.utility_batch_size

        assert trainer_config.batch_size % self.gradient_accumulation_steps == 0, \
            f"batch_size ({trainer_config.batch_size}) must be divisible by gradient_accumulation_steps ({self.gradient_accumulation_steps})"
        trainer_config.batch_size //= self.gradient_accumulation_steps
        if trainer_config.utility_batch_size is not None:
            assert trainer_config.utility_batch_size % self.gradient_accumulation_steps == 0, \
                f"utility_batch_size ({trainer_config.utility_batch_size}) must be divisible by gradient_accumulation_steps ({self.gradient_accumulation_steps})"
            trainer_config.utility_batch_size //= self.gradient_accumulation_steps
        super().__init__(trainer_config=trainer_config, **kwargs)

        if self.wandb_run is not None:
            self.wandb_run.name = "dpo2"
            self.wandb_run.config.update({
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "micro_batch_size": trainer_config.batch_size,
                "effective_batch_size": effective_batch_size,
            })

        if self.logger is not None:
            self.logger.info(
                f"Gradient accumulation: micro_batch={trainer_config.batch_size} x "
                f"accum_steps={self.gradient_accumulation_steps} = "
                f"effective_batch={effective_batch_size}"
            )
            if effective_utility_batch_size is not None:
                self.logger.info(
                    f"Utility gradient accumulation: micro_batch={trainer_config.utility_batch_size} x "
                    f"accum_steps={self.gradient_accumulation_steps} = "
                    f"effective_utility_batch={effective_utility_batch_size}"
                )

    def _build_attack_target_inputs(self, inputs):
        return inputs["harmful_ids"], inputs["harmful_mask"]

    def train(self, next_steps: int = -1):
        if next_steps < 0:
            next_steps = self.train_steps
        next_steps = min(next_steps, self.train_steps - self.current_step)

        pbar = tqdm(range(next_steps), desc="Training", initial=self.current_step, total=self.train_steps)

        for _ in pbar:
            self.current_step += 1
            self.model.train()
            self.optimizer.zero_grad()

            accum_losses = {}

            for micro_idx in range(self.gradient_accumulation_steps):
                inputs = next(self.trainloader)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                advsfx_inputs = {"ids": None, "embeds": None}
                if self.attacker is not None:
                    target_ids, target_mask = self._build_attack_target_inputs(inputs)
                    temp = self.attacker.attack_embeds(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        message_ids=inputs["prompt_ids"],
                        message_mask=inputs["prompt_mask"],
                        target_ids=target_ids,
                        target_mask=target_mask,
                        device=self.device,
                    )
                    if temp is None:
                        advsfx_ids, advsfx_embeds = None, None
                    elif len(temp.shape) == 2:
                        advsfx_ids, advsfx_embeds = temp, None
                    elif len(temp.shape) == 3:
                        advsfx_ids, advsfx_embeds = None, temp
                    else:
                        raise RuntimeError
                    advsfx_inputs = {"ids": advsfx_ids, "embeds": advsfx_embeds}

                utility_inputs = None
                if self.utilityset is not None:
                    utility_inputs = next(self.utilityloader)
                    utility_inputs = {k: v.to(self.device) for k, v in utility_inputs.items()}

                losses = self._micro_forward_backward(inputs, advsfx_inputs, utility_inputs)
                for k, v in losses.items():
                    accum_losses[k] = accum_losses.get(k, 0.0) + v

            self.optimizer.step()
            self.scheduler.step()

            for k in accum_losses:
                accum_losses[k] /= self.gradient_accumulation_steps

            pbar.set_postfix({
                "loss": f"{accum_losses.get('total_loss', 0):.4f}",
                "dpo_acc": f"{accum_losses.get('dpo_acc', 0):.2f}",
            })

            if self.wandb_run is not None:
                wandb_payload = {"step": self.current_step}
                for k, v in accum_losses.items():
                    if isinstance(v, (int, float)):
                        wandb_payload[k] = v
                self.wandb_run.log(wandb_payload, step=self.current_step)

            if (self.logger is not None) and (self.current_step % self.report_freq == 0):
                log_lines = [f"Step: [{self.current_step}/{self.train_steps}]"]
                for k, v in accum_losses.items():
                    log_lines.append("{}: {:.6f}".format(k, v))
                msg = "\n".join(log_lines)
                tqdm.write(msg)
                self.logger.info(msg)

    def _micro_forward_backward(self, inputs, advsfx_inputs, utility_inputs=None):
        scale = 1.0 / self.gradient_accumulation_steps

        context_inputs = self._build_context_inputs(inputs=inputs, advsfx_inputs=advsfx_inputs)

        chosen_ids = inputs["benign_ids"]
        chosen_mask = inputs["benign_mask"]
        rejected_ids = inputs["harmful_ids"]
        rejected_mask = inputs["harmful_mask"]

        pi_chosen_logp = self._compute_response_logp(
            model=self.model, context_inputs=context_inputs,
            response_ids=chosen_ids, response_mask=chosen_mask,
        )
        pi_rejected_logp = self._compute_response_logp(
            model=self.model, context_inputs=context_inputs,
            response_ids=rejected_ids, response_mask=rejected_mask,
        )

        with torch.no_grad():
            ref_chosen_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=chosen_ids, response_mask=chosen_mask,
            )
            ref_rejected_logp = self._compute_reference_response_logp(
                context_inputs=context_inputs,
                response_ids=rejected_ids, response_mask=rejected_mask,
            )

        delta_pi = pi_chosen_logp - pi_rejected_logp
        delta_ref = ref_chosen_logp - ref_rejected_logp
        reward_chosen = self.dpo_beta * (pi_chosen_logp - ref_chosen_logp)
        reward_rejected = self.dpo_beta * (pi_rejected_logp - ref_rejected_logp)
        reward_margin = reward_chosen - reward_rejected

        loss = -F.logsigmoid(self.dpo_beta * (delta_pi - delta_ref)).mean()
        (loss * self.adv_alpha * scale).backward()

        dpo_margin = delta_pi.mean().item()
        dpo_acc = (delta_pi > delta_ref).to(torch.float32).mean().item()
        reward_winner_rate = (reward_chosen > reward_rejected).to(torch.float32).mean().item()
        align_alpha_loss = (loss * self.adv_alpha).item()
        loss_val = loss.item()

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
                u_logps = torch.gather(u_logps[:, :-1], dim=-1, index=u_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                u_mask = u_mask[:, 1:]

                u_loss = -(u_logps * u_mask).sum() / u_cnt
                (u_loss * self.utility_alpha * scale).backward()
                utility_loss += u_loss.item()
                utility_alpha_loss += (u_loss * self.utility_alpha).item()

        return {
            "align_loss": loss_val,
            "align_alpha_loss": align_alpha_loss,
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
    "dpo2": AdvTrainerDPO2,
}


def build_trainer(trainer_config: AdvTrainerConfig, **kwargs):
    return __adv_trainer_zoo__[trainer_config.name](
        trainer_config=trainer_config,
        **kwargs,
    )
