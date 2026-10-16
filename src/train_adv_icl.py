import argparse, os, torch, numpy as np

import utils, attacks, adv_trainers


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_train_args(parser)

    parser.add_argument("--batch-size", type=int, default=None,
                        help="override trainer_cfg.batch_size if set")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--wandb-project", type=str, default=None,
                        help="if set, enable wandb logging with this project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="run name; defaults to adv-ICL-<model-basename>")

    return parser.parse_args()


def _maybe_init_wandb(args, trainer_cfg_dict, logger):
    if args.wandb_project is None:
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; skipping wandb logging. "
                       "run `pip install wandb` to enable.")
        return False

    run_name = args.wandb_run_name
    if run_name is None:
        model_name = os.path.basename(str(args.model_id).rstrip("/"))
        run_name = f"adv-ICL-{model_name}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_id": args.model_id,
            "dataset": args.dataset,
            "utilityset": args.utilityset,
            "datacollator": args.datacollator,
            "batch_size": trainer_cfg_dict.get("batch_size"),
            "gradient_accumulation_steps": trainer_cfg_dict.get("gradient_accumulation_steps"),
            "train_steps": trainer_cfg_dict.get("train_steps"),
            "random_seed": args.random_seed,
            "save_dir": args.save_dir,
            "save_name": args.save_name,
        },
    )
    logger.info(f"wandb initialized: project={args.wandb_project}, name={run_name}")
    return True


def main(args, logger):
    default_device = "cuda:0"

    lora_cfg = utils.load_yaml(args.lora_cfg_path, f"{args.save_dir}/{args.save_name}_lora-cfg.yaml")
    model, tokenizer = utils.get_model(args.model_id, default_device, lora_cfg)
    trainset = utils.get_dataset(args.dataset)

    utilityset = None
    if args.utilityset is not None:
        utilityset = utils.get_dataset(args.utilityset)

    trainer_cfg = utils.load_yaml(args.trainer_cfg_path)

    # ======== start to override trainer_cfg

    if "datacollator_config" not in trainer_cfg.keys():
        trainer_cfg["datacollator_config"] = {}
    if args.datacollator is not None:
        trainer_cfg["datacollator_config"]["name"] = args.datacollator

    if args.batch_size is not None:
        trainer_cfg["batch_size"] = args.batch_size

    trainer_cfg["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    atker_cfg = utils.load_yaml(args.atker_cfg_path)
    if atker_cfg is not None:
        trainer_cfg["attacker_config"] = atker_cfg

    # ======== finished overriding trainer_cfg

    utils.save_yaml(trainer_cfg, f"{args.save_dir}/{args.save_name}_trainer-cfg.yaml")

    wandb_active = _maybe_init_wandb(args, trainer_cfg, logger)

    trainer_cfg = adv_trainers.AdvTrainerConfig(**trainer_cfg)

    trainer = adv_trainers.build_trainer(
        trainer_config=trainer_cfg,
        model=model,
        tokenizer=tokenizer,
        device=default_device,
        trainset=trainset,
        utilityset=utilityset,
        logger=logger,
    )

    # try to resume training
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
    # utils.save_state_dict(model_state_dict, path=f"{args.save_dir}/{args.save_name}_fin-model.safetensors", mode="safetensors")

    model.save_pretrained(f"{args.save_dir}/{args.save_name}_fin-model")
    utils.save_state_dict(trainer_state_dict, path=f"{args.save_dir}/{args.save_name}_fin-trainer.pkl", mode="pytorch")

    if wandb_active:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
