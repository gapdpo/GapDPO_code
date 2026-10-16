"""Paper figures for the reward-level analysis: SFT vs DPO vs GapDPO.

Reads the per-response implicit rewards produced by
``analysis/reward_probe.py`` and emits the two figures used in the
Reward-Level Analysis subsection, restricted to the three configurations the
paper actually contrasts.

Usage
-----
    python analysis/plot_reward_separation.py
    python analysis/plot_reward_separation.py --behavior adv_training_72 --format pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "reward_probe" / "per_example_rewards.csv"
DEFAULT_OUT = HERE / "results" / "figures"

# Config name (model column of reward_probe.py) -> label used in the paper.
MODELS: list[tuple[str, str]] = [
    ("sft", "SFT"),
    ("dpo", "DPO"),
    ("gapdpo_m1k4", "GapDPO ($M{=}1$, $K{=}4$)"),
]

SAFE_FILL = "#1f8a70"
SAFE_EDGE = "#0e3f34"
HARM_FILL = "#d65a4a"
HARM_EDGE = "#6b271f"
PANEL_BG = "#fbfbf8"

RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def reward_values(row: pd.Series, prefix: str) -> list[float]:
    """Pull ``<prefix>0..N`` off one row, dropping blanks."""
    out = []
    for idx in range(16):
        key = f"{prefix}{idx}"
        if key not in row:
            break
        value = pd.to_numeric(row[key], errors="coerce")
        if pd.notna(value):
            out.append(float(value))
    return out


def load(csv_path: Path) -> pd.DataFrame:
    rewards = pd.read_csv(csv_path)
    known = {name for name, _ in MODELS}
    missing = known - set(rewards["model"])
    if missing:
        raise SystemExit(f"missing models in {csv_path}: {sorted(missing)}")
    return rewards[rewards["model"].isin(known)].copy()


def pick_behavior(rewards: pd.DataFrame, requested: str) -> str:
    """The behavior that is simultaneously typical for all three models.

    We rank each behavior within each model's own strict-gap distribution and
    take the one whose worst-case percentile deviation from the median is
    smallest, so the panel is not selected to favor any single model.
    """
    shared = set.intersection(
        *[set(group["example_id"]) for _, group in rewards.groupby("model")]
    )
    if requested != "auto":
        if requested in shared:
            return requested
        print(f"[warn] behavior={requested} not shared by all models; falling back to auto.")
    gaps = (
        rewards[rewards["example_id"].isin(shared)]
        .pivot(index="example_id", columns="model", values="min_chosen_minus_max_harmful")
    )
    deviation = (gaps.rank(pct=True) - 0.5).abs().max(axis=1)
    return str(deviation.idxmin())


def draw_panel(ax, chosen: list[float], harmful: list[float], title: str) -> float:
    """One behavior for one model: the two clusters and the strict gap between them."""
    rng = np.random.default_rng(0)
    chosen_x = 0.0 + rng.uniform(-0.13, 0.13, size=len(chosen))
    harmful_x = 1.0 + rng.uniform(-0.13, 0.13, size=len(harmful))
    chosen_min, harmful_max = min(chosen), max(harmful)
    gap = chosen_min - harmful_max

    ax.set_facecolor(PANEL_BG)
    ax.axhspan(min(chosen_min, harmful_max), max(chosen_min, harmful_max),
               color=SAFE_FILL if gap > 0 else HARM_FILL, alpha=0.11, zorder=0)

    ax.scatter(chosen_x, chosen, s=62, color=SAFE_FILL, edgecolor=SAFE_EDGE,
               linewidth=0.9, label="Safe responses", zorder=4)
    ax.scatter(harmful_x, harmful, s=62, color=HARM_FILL, marker="s", edgecolor=HARM_EDGE,
               linewidth=0.9, label="Harmful targets", zorder=4)

    ax.hlines(chosen_min, -0.30, 1.30, colors=SAFE_EDGE, linestyles="--", linewidth=1.2,
              label=r"$\min_m r^c$", zorder=3)
    ax.hlines(harmful_max, -0.30, 1.30, colors=HARM_EDGE, linestyles="--", linewidth=1.2,
              label=r"$\max_k r^h$", zorder=3)

    ax.annotate("", xy=(0.5, chosen_min), xytext=(0.5, harmful_max),
                arrowprops={"arrowstyle": "<->",
                            "color": "#2f6f4f" if gap > 0 else "#a4453a",
                            "linewidth": 1.4}, zorder=5)
    ax.annotate(
        rf"$\Delta = {gap:+.3f}$",
        xy=(0.5, (chosen_min + harmful_max) / 2), ha="center", va="center", fontsize=8.5,
        color="#234032" if gap > 0 else "#7a2f26",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "linewidth": 0.8,
              "edgecolor": "#9bc8aa" if gap > 0 else "#d9a49c"},
        zorder=6,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Safe", "Harmful"])
    ax.set_xlim(-0.45, 1.45)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title)
    return gap


def plot_cluster_grid(rewards: pd.DataFrame, behavior: str, out_path: Path) -> None:
    """The three models on a single behavior, shared y-axis."""
    panels = []
    for model, label in MODELS:
        row = rewards[(rewards["model"] == model) & (rewards["example_id"] == behavior)]
        if row.empty:
            continue
        sample = row.iloc[0]
        chosen, harmful = reward_values(sample, "chosen_reward_"), reward_values(sample, "harmful_reward_")
        if chosen and harmful:
            panels.append((label, chosen, harmful))
    if not panels:
        raise SystemExit(f"no data for behavior {behavior}")

    every = [v for _, c, h in panels for v in c + h]
    span = max(every) - min(every)
    y_lo, y_hi = min(every) - 0.16 * span - 0.02, max(every) + 0.14 * span + 0.02

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.7), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (label, chosen, harmful) in zip(np.atleast_1d(axes), panels):
        draw_panel(ax, chosen, harmful, label)
        ax.set_ylim(y_lo, y_hi)
    np.atleast_1d(axes)[0].set_ylabel("Implicit reward")

    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_all_behaviors(rewards: pd.DataFrame, out_path: Path) -> None:
    """Every response of every behavior, sorted by strict gap, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.5), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (model, label) in zip(np.atleast_1d(axes), MODELS):
        group = (rewards[rewards["model"] == model]
                 .sort_values("min_chosen_minus_max_harmful")
                 .reset_index(drop=True))
        ax.set_facecolor(PANEL_BG)
        for position, (_, sample) in enumerate(group.iterrows()):
            chosen, harmful = reward_values(sample, "chosen_reward_"), reward_values(sample, "harmful_reward_")
            if chosen:
                ax.scatter([position] * len(chosen), chosen, s=5, color=SAFE_FILL,
                           alpha=0.55, linewidths=0)
            if harmful:
                ax.scatter([position] * len(harmful), harmful, s=5, color=HARM_FILL,
                           marker="s", alpha=0.55, linewidths=0)
        separated = float((group["min_chosen_minus_max_harmful"] > 0).mean())
        ax.set_title(f"{label}\nSep. = {separated:.0%}")
        ax.set_xlabel("Behavior (sorted by $\\Delta_i$)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Implicit reward")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SAFE_FILL, markersize=5,
                   label="Safe responses"),
        plt.Line2D([], [], marker="s", linestyle="", color=HARM_FILL, markersize=5,
                   label="Harmful targets"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--behavior", default="auto",
                        help="behavior for the per-behavior panel, or 'auto' for the "
                             "one that is typical for all three models")
    parser.add_argument("--format", default="both", choices=["both", "pdf", "png"],
                        help="pdf for LaTeX, png to view or download (default: both)")
    args = parser.parse_args()

    rewards = load(args.csv)
    behavior = pick_behavior(rewards, args.behavior)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffixes = ["pdf", "png"] if args.format == "both" else [args.format]
    with plt.rc_context(RC):
        for suffix in suffixes:
            plot_cluster_grid(rewards, behavior,
                              args.out_dir / f"cluster_grid_3models.{suffix}")
            plot_all_behaviors(rewards,
                               args.out_dir / f"behavior_clusters_3models.{suffix}")

    summary = (
        rewards.groupby("model")["min_chosen_minus_max_harmful"]
        .agg(mean_gap="mean", sep=lambda s: float((s > 0).mean()))
    )
    print(f"\nbehavior shown: {behavior}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
