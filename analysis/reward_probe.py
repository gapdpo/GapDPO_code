#!/usr/bin/env python3
"""Reward-level statistics of Table 3 (and the per-response rewards behind Figure 3).

For each of the 100 adversarial-training behaviors, the probe scores the four self-generated
refusals (styles i-iv) and the first four corpus affirmative targets with the implicit reward

    r(x, y) = beta * ( mean_t log pi_theta(y_t | x, y_<t) - mean_t log pi_ref(y_t | x, y_<t) ),

i.e. length-normalized log-probabilities, beta = 0.3, and pi_ref = the policy with its LoRA adapter
disabled. Prompts are the clean behaviors in the Qwen chat template (no adversarial suffix).

    python analysis/reward_probe.py --gpu 0                       # Table 3 rows, adapters from results/<config>
    python analysis/reward_probe.py --gpu 0 --adapter dpo=/path/to/dpo_fin-model ...

Writes analysis/results/reward_probe/{per_example_rewards.csv, table3.csv} and prints Table 3.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path


def _preparse_gpu() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="0")
    known, _ = parser.parse_known_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(known.gpu))


_preparse_gpu()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/harmbench"
DEFAULT_OUT_DIR = ROOT / "analysis/results/reward_probe"
# config name (scripts/train/run_paper_config.sh) -> Table 3 label
TABLE3 = [
    ("sft", "SFT"),
    ("adv_icl", "Adv-ICL [27]"),
    ("dpo", "DPO [8]"),
    ("gapdpo_wo_vh_m1k4", "GapDPO w/o Vh (M=1,K=4)"),
    ("gapdpo_m1k4", "GapDPO (M=1,K=4)"),
]
BETA = 0.3
NUM_CHOSEN = 4
NUM_HARMFUL = 4

QWEN_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN_TARGET_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"


def load_behaviors() -> list[dict]:
    with (DATA / "behavior_datasets/extra_behavior_datasets/adv_training_behaviors_chosen4.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = json.loads((DATA / "optimizer_targets/extra_targets/adv_training_targets_chosen4.json").read_text())
    harmful = json.loads((DATA / "optimizer_targets/extra_targets/adv_training_targets.json").read_text())
    return [{"example_id": r["BehaviorID"], "prompt": r["Behavior"],
             "chosen": list(chosen[r["BehaviorID"]])[:NUM_CHOSEN],
             "harmful": list(harmful[r["BehaviorID"]])[:NUM_HARMFUL]} for r in rows]


def encode_prompt(tokenizer, prompt: str):
    ids = tokenizer.encode(prompt.strip(), add_special_tokens=False, return_tensors="pt")[0]
    prefix = tokenizer.encode(QWEN_PROMPT_PREFIX, add_special_tokens=True, return_tensors="pt")[0]
    ids = torch.cat([prefix, ids])
    return ids, torch.ones_like(ids)


def encode_response(tokenizer, response: str):
    ids = tokenizer.encode(response.strip(), add_special_tokens=False, return_tensors="pt")[0]
    prefix = tokenizer.encode(QWEN_TARGET_PREFIX, add_special_tokens=False, return_tensors="pt")[0]
    ids = torch.cat([prefix, ids])
    mask = torch.cat([torch.zeros_like(prefix), torch.ones(len(ids) - len(prefix), dtype=torch.int64)])
    return ids, mask


def pad(items, pad_id: int, left: bool):
    max_len = max(len(ids) for ids, _ in items)
    ids_out = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    mask_out = torch.zeros_like(ids_out)
    for i, (ids, mask) in enumerate(items):
        if left:
            ids_out[i, -len(ids):], mask_out[i, -len(mask):] = ids, mask
        else:
            ids_out[i, :len(ids)], mask_out[i, :len(mask)] = ids, mask
    return ids_out, mask_out


def response_logp(model, context_ids, context_mask, response_ids, response_mask):
    input_ids = torch.cat([context_ids, response_ids], dim=1)
    attention_mask = torch.cat([context_mask, torch.ones_like(response_ids)], dim=1)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, -(response_ids.shape[-1] + 1):-1]
    token_logps = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
    token_mask = response_mask.to(token_logps.dtype)
    return (token_logps * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)


def score_group(model, tokenizer, batch: list[dict], key: str, device) -> list[list[float]]:
    pad_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]
    contexts, responses, sizes = [], [], []
    for b in batch:
        sizes.append(len(b[key]))
        for response in b[key]:
            contexts.append(encode_prompt(tokenizer, b["prompt"]))
            responses.append(encode_response(tokenizer, response))
    c_ids, c_mask = (t.to(device) for t in pad(contexts, pad_id, left=True))
    r_ids, r_mask = (t.to(device) for t in pad(responses, pad_id, left=False))
    with torch.no_grad():
        pi = response_logp(model, c_ids, c_mask, r_ids, r_mask)
        with model.disable_adapter():
            ref = response_logp(model, c_ids, c_mask, r_ids, r_mask)
    flat = (BETA * (pi - ref)).cpu().tolist()
    out, i = [], 0
    for n in sizes:
        out.append(flat[i:i + n])
        i += n
    return out


def score_model(name: str, adapter: Path, args, behaviors: list[dict]) -> list[dict]:
    print(f"Scoring {name}: {adapter}", flush=True)
    device = torch.device("cuda:0")
    base = AutoModelForCausalLM.from_pretrained(str(args.model_id), torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_id))
    rows = []
    for start in range(0, len(behaviors), args.batch_size):
        batch = behaviors[start:start + args.batch_size]
        chosen = score_group(model, tokenizer, batch, "chosen", device)
        harmful = score_group(model, tokenizer, batch, "harmful", device)
        for b, c, h in zip(batch, chosen, harmful):
            row = {"model": name, "example_id": b["example_id"], "prompt": b["prompt"]}
            row.update({f"chosen_reward_{i}": v for i, v in enumerate(c)})
            row.update({f"harmful_reward_{i}": v for i, v in enumerate(h)})
            row["min_chosen_minus_max_harmful"] = min(c) - max(h)
            rows.append(row)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def table3(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    chosen = df[[f"chosen_reward_{i}" for i in range(NUM_CHOSEN)]]
    harmful = df[[f"harmful_reward_{i}" for i in range(NUM_HARMFUL)]]
    per = pd.DataFrame({
        "model": df["model"],
        "mu_c": chosen.mean(axis=1), "V_c": chosen.var(axis=1, ddof=0), "min_r_c": chosen.min(axis=1),
        "mu_h": harmful.mean(axis=1), "V_h": harmful.var(axis=1, ddof=0), "max_r_h": harmful.max(axis=1),
    })
    per["Delta"] = per["min_r_c"] - per["max_r_h"]
    per["Sep"] = 100.0 * (per["Delta"] > 0)
    out = per.groupby("model", sort=False).mean().reset_index()
    out.insert(1, "label", out["model"].map(labels).fillna(out["model"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-id", type=Path, default=ROOT / "models/Qwen2.5-7B-Instruct")
    parser.add_argument("--configs", nargs="+", default=[c for c, _ in TABLE3],
                        help="config names; adapters default to results/<config>/<config>_fin-model")
    parser.add_argument("--adapter", action="append", default=[], metavar="CONFIG=PATH",
                        help="override the adapter directory of one config")
    parser.add_argument("--batch-size", type=int, default=2, help="behaviors per forward pass (Table 3 used 2)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(item.split("=", 1) for item in args.adapter)
    behaviors = load_behaviors()
    rows = []
    for config in args.configs:
        adapter = Path(overrides.get(config, ROOT / "results" / config / f"{config}_fin-model"))
        if not adapter.exists():
            raise SystemExit(f"missing adapter for {config}: {adapter}")
        rows.extend(score_model(config, adapter, args, behaviors))
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_example_rewards.csv", index=False)
    table = table3(df, dict(TABLE3))
    table.to_csv(args.out_dir / "table3.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
