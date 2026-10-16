import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def candidate_model_roots():
    roots = []
    model_root = os.getenv("MODEL_ROOT")
    if model_root:
        roots.append(Path(model_root).expanduser())
    roots.append(REPO_ROOT / "models")
    roots.append(Path.home() / "models")

    deduped = []
    seen = set()
    for root in roots:
        root = root.resolve(strict=False)
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def resolve_model_path(model_name: str) -> str:
    for root in candidate_model_roots():
        candidate = root / model_name
        if candidate.exists():
            return str(candidate)
    return str(candidate_model_roots()[0] / model_name)
