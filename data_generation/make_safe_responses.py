import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List
from difflib import SequenceMatcher

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = str(REPO_ROOT / "models/Qwen2.5-7B-Instruct")
DEFAULT_DATA_ROOT = REPO_ROOT / "data/harmbench"
DEFAULT_BEHAVIOR_PATH = DEFAULT_DATA_ROOT / "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
DEFAULT_REJECTED_PATH = DEFAULT_DATA_ROOT / "optimizer_targets/extra_targets/adv_training_targets.json"
DEFAULT_OUTPUT_BEHAVIOR_PATH = DEFAULT_DATA_ROOT / "behavior_datasets/extra_behavior_datasets/adv_training_behaviors_chosen4.csv"
DEFAULT_OUTPUT_CHOSEN_PATH = DEFAULT_DATA_ROOT / "optimizer_targets/extra_targets/adv_training_targets_chosen4.json"
DEFAULT_OUTPUT_PREVIEW_PATH = DEFAULT_DATA_ROOT / "optimizer_targets/extra_targets/adv_training_targets_chosen4_preview.csv"

# Keep the styles fixed so the resulting chosen set is intentionally diverse.
STYLE_SPECS = [
    {
        "style_id": "firm_refusal",
        "style_name": "Firm Refusal",
        "min_words": 5,
        "instruction": (
            "Return exactly 1 sentence. The sentence must begin with "
            "\"I can't\" or \"I cannot\" and only give a firm refusal. "
            "Do not offer alternatives, empathy, or extra explanation."
        ),
    },
    {
        "style_id": "policy_boundary",
        "style_name": "Policy Boundary",
        "min_words": 8,
        "instruction": (
            "Return exactly 2 sentences. The first sentence must be a direct refusal. "
            "The second sentence must briefly explain the safety, legal, ethical, "
            "privacy, or security boundary. Do not offer alternatives."
        ),
    },
    {
        "style_id": "safe_redirect",
        "style_name": "Safe Redirect",
        "min_words": 8,
        "instruction": (
            "Return exactly 2 sentences. The first sentence must be a direct refusal. "
            "The second sentence must start with \"If you'd like,\" and redirect "
            "the user to a harmless, lawful, preventive, or high-level alternative."
        ),
    },
    {
        "style_id": "empathetic_boundary",
        "style_name": "Empathetic Boundary",
        "min_words": 8,
        "instruction": (
            "Return exactly 2 sentences. The first sentence must start with "
            "\"I understand\" and include a refusal. The second sentence must start "
            "with \"I can help\" and offer safe, legitimate help."
        ),
    },
]

BOUNDARY_WORDS = (
    "illegal",
    "legal",
    "safety",
    "safe",
    "harmful",
    "policy",
    "policies",
    "ethical",
    "ethics",
    "privacy",
    "security",
    "unsafe",
)

REDIRECT_WORDS = (
    "safe",
    "safely",
    "lawful",
    "legal",
    "prevent",
    "prevention",
    "protect",
    "protection",
    "secure",
    "security",
    "best practices",
    "high-level",
)

DISALLOWED_PATTERNS = (
    "```",
    "http://",
    "https://",
    "step 1",
    "1.",
    "2.",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BAD_PUNCT_SPACING_RE = re.compile(r"(?<=[a-zA-Z])\.(?=[A-Z])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four style-diverse chosen responses per HarmBench behavior for DPO training."
    )
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--behavior-path", type=Path, default=DEFAULT_BEHAVIOR_PATH)
    parser.add_argument("--rejected-path", type=Path, default=DEFAULT_REJECTED_PATH)
    parser.add_argument("--output-behavior-path", type=Path, default=DEFAULT_OUTPUT_BEHAVIOR_PATH)
    parser.add_argument("--output-chosen-path", type=Path, default=DEFAULT_OUTPUT_CHOSEN_PATH)
    parser.add_argument("--output-preview-path", type=Path, default=DEFAULT_OUTPUT_PREVIEW_PATH)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-attempts-per-style", type=int, default=6)
    return parser.parse_args()


def load_behaviors(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rejected(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_generation_prompt(
    behavior: str,
    style_spec: Dict[str, str],
    avoid_texts: List[str],
    attempt_idx: int,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You are writing preferred safe assistant responses for safety DPO training. "
        "Produce safe refusals that differ strongly in tone and structure while remaining harmless. "
        "Follow the requested structure exactly."
    )
    user_prompt = (
        "Write a safe assistant response to the harmful request below.\n\n"
        "Style goal:\n"
        f"- {style_spec['style_name']}: {style_spec['instruction']}\n\n"
        "Hard requirements:\n"
        "1. The first sentence must clearly refuse the request.\n"
        "2. Use at most 2 sentences total, and follow the style's exact sentence count.\n"
        "3. Keep the response between 8 and 70 words.\n"
        "4. Do not provide any harmful, illegal, violent, or procedural details.\n"
        "5. Do not include code, steps, examples, links, placeholders, or roleplay.\n"
        "6. Keep this style clearly distinct from the other styles.\n"
        "7. Output only the assistant response text.\n\n"
        f"User request: {behavior}"
    )
    if avoid_texts:
        formatted = "\n".join(f"- {text}" for text in avoid_texts[-4:])
        user_prompt += (
            "\n\nDo not produce a response too similar to these prior responses:\n"
            f"{formatted}"
        )
    if attempt_idx > 1:
        user_prompt += (
            "\n\nRetry note: the previous draft was too generic, too similar, or did not follow the required style. "
            "Make this response more distinct while keeping it safe."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def decode_response(tokenizer, outputs, prompt_len: int) -> str:
    gen_ids = outputs[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return " ".join(text.split())


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip()) if part.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_refusal_sentence(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "i can't",
        "i cannot",
        "i can’t",
        "i won't",
        "i will not",
        "sorry, but i can't",
        "sorry, but i cannot",
        "unfortunately, i can't",
        "unfortunately, i cannot",
    )
    return any(marker in lowered for marker in markers)


def has_any_word(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def validate_common_response(text: str, min_words: int) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if BAD_PUNCT_SPACING_RE.search(text):
        return False, "bad_punctuation_spacing"
    if any(pattern in text.lower() for pattern in DISALLOWED_PATTERNS):
        return False, "disallowed_pattern"
    wc = word_count(text)
    if wc < min_words:
        return False, "too_short"
    if wc > 70:
        return False, "too_long"
    sentences = split_sentences(text)
    if len(sentences) == 0 or len(sentences) > 2:
        return False, "bad_sentence_count"
    if not is_refusal_sentence(sentences[0]):
        return False, "missing_refusal_first_sentence"
    return True, "ok"


def validate_style_response(style_spec: Dict[str, str], text: str) -> tuple[bool, str]:
    ok, reason = validate_common_response(text, min_words=style_spec["min_words"])
    if not ok:
        return ok, reason

    style_id = style_spec["style_id"]
    sentences = split_sentences(text)
    first = sentences[0]
    second = sentences[1] if len(sentences) > 1 else ""
    lowered = text.lower()

    if style_id == "firm_refusal":
        if len(sentences) != 1:
            return False, "firm_refusal_requires_one_sentence"
        if not lowered.startswith("i can't") and not lowered.startswith("i cannot"):
            return False, "firm_refusal_requires_direct_opening"
        if "if you'd like" in lowered or lowered.startswith("i understand"):
            return False, "firm_refusal_too_similar_to_other_styles"
        return True, "ok"

    if style_id == "policy_boundary":
        if len(sentences) != 2:
            return False, "policy_boundary_requires_two_sentences"
        if has_any_word(second, ("if you'd like", "i can help", "i understand")):
            return False, "policy_boundary_should_not_redirect"
        if not has_any_word(second, BOUNDARY_WORDS):
            return False, "policy_boundary_missing_boundary_reason"
        return True, "ok"

    if style_id == "safe_redirect":
        if len(sentences) != 2:
            return False, "safe_redirect_requires_two_sentences"
        if not second.lower().startswith("if you'd like"):
            return False, "safe_redirect_requires_if_youd_like"
        if not has_any_word(second, REDIRECT_WORDS):
            return False, "safe_redirect_missing_safe_alternative"
        return True, "ok"

    if style_id == "empathetic_boundary":
        if len(sentences) != 2:
            return False, "empathetic_boundary_requires_two_sentences"
        if not first.lower().startswith("i understand"):
            return False, "empathetic_boundary_requires_i_understand"
        if not second.lower().startswith("i can help"):
            return False, "empathetic_boundary_requires_i_can_help"
        return True, "ok"

    return False, f"unknown_style_{style_id}"


def is_too_similar(candidate: str, existing: List[str], threshold: float = 0.88) -> bool:
    normalized = normalize_for_compare(candidate)
    for prev in existing:
        prev_norm = normalize_for_compare(prev)
        if normalized == prev_norm:
            return True
        if SequenceMatcher(None, normalized, prev_norm).ratio() >= threshold:
            return True
    return False


def build_fallback_response(style_spec: Dict[str, str]) -> str:
    style_id = style_spec["style_id"]

    if style_id == "firm_refusal":
        return "I cannot assist with that request."
    if style_id == "policy_boundary":
        return "I cannot assist with that request. It violates safety and legal boundaries."
    if style_id == "safe_redirect":
        return "I cannot assist with that request. If you'd like, I can help with safe, lawful alternatives instead."
    if style_id == "empathetic_boundary":
        return "I understand you're asking about that, but I can't assist with it. I can help with safe, legitimate alternatives instead."

    raise ValueError(f"unknown_style_{style_id}")


def generate_chosen_response(
    model,
    tokenizer,
    behavior: str,
    style_spec: Dict[str, str],
    avoid_texts: List[str],
    attempt_idx: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    chat = build_generation_prompt(behavior, style_spec, avoid_texts, attempt_idx)
    input_ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    sample_temperature = temperature
    if temperature > 0:
        sample_temperature = min(0.7, temperature + 0.05 * max(0, attempt_idx - 1))

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=sample_temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return decode_response(tokenizer, outputs, input_ids.shape[-1])


def select_style_response(
    model,
    tokenizer,
    behavior: str,
    style_spec: Dict[str, str],
    existing_choices: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_attempts: int,
) -> str:
    attempted: List[str] = []
    last_candidate = ""
    last_reason = "unknown"

    for attempt_idx in range(1, max_attempts + 1):
        avoid_texts = existing_choices + attempted
        candidate = generate_chosen_response(
            model=model,
            tokenizer=tokenizer,
            behavior=behavior,
            style_spec=style_spec,
            avoid_texts=avoid_texts,
            attempt_idx=attempt_idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ).strip()

        last_candidate = candidate
        attempted.append(candidate)

        ok, reason = validate_style_response(style_spec, candidate)
        if not ok:
            last_reason = reason
            continue

        if is_too_similar(candidate, existing_choices):
            last_reason = "too_similar_to_existing_style"
            continue

        return candidate

    fallback = build_fallback_response(style_spec)
    ok, reason = validate_style_response(style_spec, fallback)
    if not ok:
        raise RuntimeError(
            f"Internal fallback response is invalid for style={style_spec['style_id']}: reason={reason}"
        )
    if is_too_similar(fallback, existing_choices):
        raise RuntimeError(
            f"Fallback response for style={style_spec['style_id']} is too similar to existing choices."
        )

    print(
        f"[warn] template fallback for style={style_spec['style_id']} after {max_attempts} attempts; "
        f"last_reason={last_reason}; last_candidate={last_candidate}; fallback={fallback}",
        file=sys.stderr,
    )
    return fallback


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


def save_preview(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "BehaviorID",
        "Behavior",
        "style_1",
        "style_2",
        "style_3",
        "style_4",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    import os
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
    preview_rows: List[Dict[str, str]] = []

    iterator = selected_rows if tqdm is None else tqdm(selected_rows, desc="Generating chosen x4", unit="sample")
    for idx, row in enumerate(iterator, start=1):
        behavior_id = row["BehaviorID"]
        behavior = row["Behavior"]
        if behavior_id not in rejected:
            raise KeyError(f"Missing rejected target for BehaviorID={behavior_id}")

        chosen_list: List[str] = []
        preview_row = {
            "BehaviorID": behavior_id,
            "Behavior": behavior,
        }

        for style_idx, style_spec in enumerate(STYLE_SPECS, start=1):
            chosen = select_style_response(
                model=model,
                tokenizer=tokenizer,
                behavior=behavior,
                style_spec=style_spec,
                existing_choices=chosen_list,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_attempts=args.max_attempts_per_style,
            )
            chosen_list.append(chosen)
            preview_row[f"style_{style_idx}"] = chosen

        chosen_payload[behavior_id] = chosen_list
        preview_rows.append(preview_row)

        print(f"[{idx}/{len(selected_rows)}] {behavior_id}")
        for style_idx, chosen in enumerate(chosen_list, start=1):
            print(f"style_{style_idx}: {chosen}")
        print()

    save_behaviors(args.output_behavior_path, selected_rows)
    save_chosen(args.output_chosen_path, chosen_payload)
    save_preview(args.output_preview_path, preview_rows)

    print(f"saved behavior subset to {args.output_behavior_path}")
    print(f"saved chosen targets to {args.output_chosen_path}")
    print(f"saved preview csv to {args.output_preview_path}")
    print(f"source rejected targets remain at {args.rejected_path}")


if __name__ == "__main__":
    main()
