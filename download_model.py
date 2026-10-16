"""Download multiple Hugging Face models in one run."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_IDS = [
    #"meta-llama/Llama-2-7b-chat-hf",
    #"meta-llama/Meta-Llama-3-8B-Instruct",
    #"Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "cais/HarmBench-Llama-2-13b-cls",
    "osunlp/AmpleGCG-plus-llama2-sourced-vicuna-7b13b-guanaco-7b13b"
]


def download_models(save_root: Path) -> None:
    failed_models = []

    for model_id in MODEL_IDS:
        local_dir = save_root / model_id.split("/")[-1]
        print(f"[START] {model_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"[DONE]  {model_id}\n")
        except Exception as exc:
            print(f"[FAIL]  {model_id}: {exc}\n")
            failed_models.append(model_id)

    if failed_models:
        print("Failed models:")
        for model_id in failed_models:
            print(f" - {model_id}")
    else:
        print("All models downloaded successfully.")


if __name__ == "__main__":
    save_dir = Path("./models")
    save_dir.mkdir(parents=True, exist_ok=True)
    download_models(save_dir)