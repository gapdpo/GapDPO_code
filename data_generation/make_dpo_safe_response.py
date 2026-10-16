import argparse
import csv
import json
from pathlib import Path
import torch
from typing import Dict, List

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = str(REPO_ROOT / "models/Qwen2.5-7B-Instruct")
DEFAULT_DATA_ROOT = REPO_ROOT / "data/harmbench"
DEFAULT_BEHAVIOR_PATH = DEFAULT_DATA_ROOT / "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
DEFAULT_REJECTED_PATH = DEFAULT_DATA_ROOT / "optimizer_targets/extra_targets/adv_training_targets.json"
DEFAULT_OUTPUT_BEHAVIOR_PATH = DEFAULT_DATA_ROOT / "behavior_datasets/extra_behavior_datasets/adv_training_behaviors_chosen.csv"
DEFAULT_OUTPUT_CHOSEN_PATH = DEFAULT_DATA_ROOT / "optimizer_targets/extra_targets/adv_training_targets_chosen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompt-specific chosen responses for HarmBench.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--behavior-path", type=Path, default=DEFAULT_BEHAVIOR_PATH)
    parser.add_argument("--rejected-path", type=Path, default=DEFAULT_REJECTED_PATH)
    parser.add_argument("--output-behavior-path", type=Path, default=DEFAULT_OUTPUT_BEHAVIOR_PATH)
    parser.add_argument("--output-chosen-path", type=Path, default=DEFAULT_OUTPUT_CHOSEN_PATH)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=1)
    return parser.parse_args()


def load_behaviors(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rejected(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_generation_prompt(behavior: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are writing preferred safe assistant responses for safety DPO training. "
        "Produce concise, firm refusals to harmful requests."
    )
    user_prompt = (
        "Write a safe assistant response to the harmful request below.\n\n"
        "Requirements:\n"
        "1. The first sentence must clearly refuse the request.\n"
        "2. Use at most 2 sentences total.\n"
        "3. Keep the response under 60 words.\n"
        "4. Do not provide any harmful, illegal, violent, or procedural details.\n"
        "5. Do not mention links, unavailable resources, or vague placeholders like 'here'.\n"
        "6. Optionally add one short safe redirection sentence.\n"
        "7. Output only the assistant response text.\n\n"
        f"User request: {behavior}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_chosen_response(model, tokenizer, behavior: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    chat = build_generation_prompt(behavior)
    input_ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def save_behaviors(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_chosen(path: Path, payload: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.manual_seed(args.seed)

    behaviors = load_behaviors(args.behavior_path)
    rejected = load_rejected(args.rejected_path)
    selected_rows = behaviors if args.num_samples is None else behaviors[: args.num_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    chosen_payload: Dict[str, List[str]] = {}
    iterator = selected_rows if tqdm is None else tqdm(selected_rows, desc="Generating chosen", unit="sample")
    for idx, row in enumerate(iterator, start=1):
        behavior_id = row["BehaviorID"]
        behavior = row["Behavior"]
        if behavior_id not in rejected:
            raise KeyError(f"Missing rejected target for BehaviorID={behavior_id}")

        chosen = generate_chosen_response(
            model=model,
            tokenizer=tokenizer,
            behavior=behavior,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        chosen_payload[behavior_id] = [chosen]
        print(f"[{idx}/{len(selected_rows)}] {behavior_id}\nchosen: {chosen}\n")

    save_behaviors(args.output_behavior_path, selected_rows)
    save_chosen(args.output_chosen_path, chosen_payload)

    print(f"saved behavior subset to {args.output_behavior_path}")
    print(f"saved chosen targets to {args.output_chosen_path}")
    print(f"source rejected targets remain at {args.rejected_path}")


if __name__ == "__main__":
    main()
