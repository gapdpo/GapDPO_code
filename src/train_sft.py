import argparse
import copy
import importlib
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils


def preload_external_wandb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.path.abspath(os.getcwd())
    original_path = list(sys.path)
    sys.path = [
        path for path in sys.path
        if os.path.abspath(path or cwd) != script_dir
    ]
    try:
        importlib.import_module("wandb")
    except Exception:
        sys.modules.pop("wandb", None)
    finally:
        sys.path = original_path


preload_external_wandb()

from adv_trainers import trainer_sft as trainer_impl
from adv_trainers.data import Loader
from utils.datasets import HarmBench
from utils.models import load_peft_state_dict, named_peft_parameters, peft_state_dict


class DistributedAwarePEFTWrapper:
    def __init__(self, model, peft_name: str = None):
        self.model = model
        self.peft_name = peft_name

    def _unwrap(self):
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    @property
    def vocab_size(self):
        model = self._unwrap()
        if hasattr(model, "vocab_size"):
            return model.vocab_size
        return model.config.vocab_size

    def get_input_embeddings(self):
        return self._unwrap().get_input_embeddings()

    def generate(self, *args, **kwargs):
        return self._unwrap().generate(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self._unwrap().save_pretrained(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def named_parameters(self):
        return named_peft_parameters(self._unwrap(), self.peft_name)

    def parameters(self):
        for _, pp in self.named_parameters():
            yield pp

    def state_dict(self) -> dict:
        return peft_state_dict(self._unwrap(), self.peft_name)

    def load_state_dict(self, state_dict):
        load_peft_state_dict(self._unwrap(), self.peft_name, state_dict)


class DistributedLoopingLoader(Loader):
    def __init__(self, dataset, batch_size, collate_fn, rank, world_size, seed=0):
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        self.iterator = None
        self.epoch = 0

    def __next__(self):
        if self.iterator is None:
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples


class HarmBenchChosen4SingleIndex(HarmBench):
    def __init__(self, root: str = "../data", chosen_index: int = 2):
        super().__init__(root=root, split="train")

        chosen_path = "harmbench/optimizer_targets/extra_targets/adv_training_targets_chosen4.json"
        self.chosen_path = os.path.join(root, chosen_path)
        with open(self.chosen_path, "rb") as f:
            self.chosen_ds = json.load(f)

        self.fixed_chosen_indices = (chosen_index,)
        self.sample_index = [
            (prompt_idx, chosen_index)
            for prompt_idx in range(len(self.prompt_ds))
        ]

        for behavior_id, chosen_list in self.chosen_ds.items():
            if len(chosen_list) <= chosen_index:
                raise ValueError(
                    f"behavior_id={behavior_id} has only {len(chosen_list)} chosen responses; "
                    f"cannot use fixed index {chosen_index}"
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


def distributed_is_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if not distributed_is_enabled():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if not is_main_process():
        os.environ["WANDB_MODE"] = "disabled"
    return torch.device(f"cuda:{local_rank}")


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_logger(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if is_main_process():
        return utils.generic_init(args)

    logger = logging.getLogger(f"rank{get_rank()}")
    logger.addHandler(logging.NullHandler())
    logger.disabled = True
    return logger


def get_model(model_id: str, device: torch.device, lora_cfg: dict = None):
    device_map = {"": str(device)} if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if lora_cfg is not None:
        lora_cfg = copy.deepcopy(lora_cfg)
        adapter_name = lora_cfg.pop("adapter_name", "default")
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
        model = DistributedAwarePEFTWrapper(model, adapter_name)

    return model, tokenizer


def maybe_wrap_ddp(model, device):
    if distributed_is_enabled():
        model.model = DDP(
            model.model,
            device_ids=[device.index],
            output_device=device.index,
        )
    return model


def replace_loaders_for_ddp(trainer, trainset, utilityset, seed):
    if not distributed_is_enabled():
        return

    rank = get_rank()
    world_size = int(os.environ["WORLD_SIZE"])
    seed = 0 if seed is None else seed

    train_batch_size = trainer.trainloader.loader.batch_size
    trainer.trainloader = DistributedLoopingLoader(
        dataset=trainset,
        batch_size=train_batch_size,
        collate_fn=trainer.train_datacollator,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    if utilityset is not None:
        utility_batch_size = trainer.utilityloader.loader.batch_size
        trainer.utilityloader = DistributedLoopingLoader(
            dataset=utilityset,
            batch_size=utility_batch_size,
            collate_fn=trainer.utility_datacollator,
            rank=rank,
            world_size=world_size,
            seed=seed + 17,
        )


def scale_batch_sizes_for_ddp(trainer_cfg, logger, gradient_accumulation_steps):
    if not distributed_is_enabled():
        return

    world_size = int(os.environ["WORLD_SIZE"])
    original_batch_size = trainer_cfg.get("batch_size")
    original_utility_batch_size = trainer_cfg.get("utility_batch_size")
    for key in ("batch_size", "utility_batch_size"):
        value = trainer_cfg.get(key)
        if value is None:
            continue
        if value % world_size != 0:
            raise ValueError(
                f"{key} ({value}) must be divisible by WORLD_SIZE ({world_size}) "
                "for DDP batch splitting"
            )
        trainer_cfg[key] = value // world_size
        if trainer_cfg[key] % gradient_accumulation_steps != 0:
            raise ValueError(
                f"per-rank {key} ({trainer_cfg[key]}) must be divisible by "
                f"gradient_accumulation_steps ({gradient_accumulation_steps})"
            )

    if is_main_process():
        per_rank_micro_batch_size = trainer_cfg["batch_size"] // gradient_accumulation_steps
        per_rank_micro_utility_batch_size = None
        if trainer_cfg.get("utility_batch_size") is not None:
            per_rank_micro_utility_batch_size = (
                trainer_cfg["utility_batch_size"] // gradient_accumulation_steps
            )
        logger.info(
            "DDP batch split: WORLD_SIZE=%d, per-rank batch_size=%s, "
            "per-rank utility_batch_size=%s",
            world_size,
            trainer_cfg.get("batch_size"),
            trainer_cfg.get("utility_batch_size"),
        )
        logger.info(
            "DDP global effective batch: train=%s "
            "(per_rank_micro=%s x accum=%s x world_size=%s)",
            original_batch_size,
            per_rank_micro_batch_size,
            gradient_accumulation_steps,
            world_size,
        )
        if original_utility_batch_size is not None:
            logger.info(
                "DDP global effective utility batch: utility=%s "
                "(per_rank_micro=%s x accum=%s x world_size=%s)",
                original_utility_batch_size,
                per_rank_micro_utility_batch_size,
                gradient_accumulation_steps,
                world_size,
            )


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)
    utils.add_train_args(parser)
    parser.add_argument("--chosen-index", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def get_trainset(args, logger):
    if args.dataset == "harmbench-chosen4":
        trainset = HarmBenchChosen4SingleIndex(chosen_index=args.chosen_index)
        if is_main_process():
            logger.info(
                "Using fixed chosen4 index %d; trainset has %d base behaviors and %d samples",
                args.chosen_index,
                len(trainset.prompt_ds),
                len(trainset),
            )
        return trainset

    return utils.get_dataset(args.dataset)


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
    trainer_cfg["name"] = "sft"
    trainer_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    trainer_cfg["wandb_run_name"] = f"sft-idx{args.chosen_index:02d}"
    scale_batch_sizes_for_ddp(trainer_cfg, logger, args.gradient_accumulation_steps)

    if args.dataset == "harmbench-chosen4":
        base_train_steps = trainer_cfg["train_steps"]
        trainer_cfg["train_steps"] = int(base_train_steps * len(trainset.fixed_chosen_indices))
        if is_main_process():
            logger.info(
                "Using fixed chosen4 index subset %s; overriding train_steps from %d to %d",
                trainset.fixed_chosen_indices,
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

    if is_main_process():
        utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")

    trainer_cfg = trainer_impl.AdvTrainerConfig(**trainer_cfg)

    trainer = trainer_impl.build_trainer(
        trainer_config=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        trainset=trainset,
        utilityset=utilityset,
        logger=logger if is_main_process() else None,
    )
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
