import argparse
import json
import os

import numpy as np
import torch

import utils
from adv_trainers import trainer_dpo3 as trainer_impl
from utils.datasets import HarmBench, resolve_data_root


class HarmBenchChosen4Fixed02(HarmBench):
    def __init__(self, root: str = "../data"):
        root = resolve_data_root(root)
        super().__init__(root=root, split="train")

        chosen_path = "harmbench/optimizer_targets/extra_targets/adv_training_targets_chosen4.json"
        self.chosen_path = os.path.join(self.root, chosen_path)
        with open(self.chosen_path, "rb") as f:
            self.chosen_ds = json.load(f)

        self.fixed_chosen_indices = (0, 2)
        self.sample_index = [
            (prompt_idx, chosen_idx)
            for prompt_idx in range(len(self.prompt_ds))
            for chosen_idx in self.fixed_chosen_indices
        ]

        for behavior_id, chosen_list in self.chosen_ds.items():
            if len(chosen_list) <= max(self.fixed_chosen_indices):
                raise ValueError(
                    f"behavior_id={behavior_id} has only {len(chosen_list)} chosen responses; "
                    f"cannot use fixed indices {self.fixed_chosen_indices}"
                )

    def __getitem__(self, idx):
        prompt_idx, chosen_idx = self.sample_index[idx]
        sample = self.prompt_ds[prompt_idx]
        behavior_id = sample["BehaviorID"]
        behavior_list = self.target_ds[behavior_id]
        chosen_list = self.chosen_ds[behavior_id]
        return {
            "prompt": sample["Behavior"],
            "target": behavior_list[np.random.randint(len(behavior_list))],
            "chosen": chosen_list[chosen_idx],
        }

    def __len__(self):
        return len(self.sample_index)


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)
    utils.add_train_args(parser)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def get_trainset(args, logger):
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


def main(args, logger):
    default_device = "cuda:0"

    lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    trainset = get_trainset(args, logger)

    utilityset = None
    if args.utilityset is not None:
        utilityset = utils.get_dataset(args.utilityset)

    trainer_cfg = utils.load_yaml(args.trainer_cfg_path)
    trainer_cfg["name"] = "dpo3"
    trainer_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    trainer_cfg["chosen_sft_alpha"] = 0.1
    trainer_cfg["wandb_run_name"] = f"dpo-260428-beta-{trainer_cfg['dpo_beta']}"

    if args.dataset == "harmbench-chosen4":
        base_train_steps = trainer_cfg["train_steps"]
        trainer_cfg["train_steps"] = int(base_train_steps * 2)
        logger.info(
            "Using fixed chosen4 index subset [0, 2]; "
            f"overriding train_steps from {base_train_steps} to {trainer_cfg['train_steps']}"
        )

    if "datacollator_config" not in trainer_cfg.keys():
        trainer_cfg["datacollator_config"] = {}
    if args.datacollator is not None:
        trainer_cfg["datacollator_config"]["name"] = args.datacollator

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        trainer_cfg["attacker_config"] = atker_cfg

    utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")

    trainer_cfg = trainer_impl.AdvTrainerConfig(**trainer_cfg)

    trainer = trainer_impl.build_trainer(
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

    model_state_dict, trainer_state_dict = trainer.state_dict()
    model.save_pretrained(f"{args.save_dir}/{args.save_name}_fin-model")
    utils.save_state_dict(
        trainer_state_dict,
        path=f"{args.save_dir}/{args.save_name}_fin-trainer.pkl",
        mode="pytorch",
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
