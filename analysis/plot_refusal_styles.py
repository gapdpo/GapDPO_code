"""Figure: how the four safe-response styles are generated.

Every style shares one prompt template; only the style-instruction slot inside
it changes. This script pulls ``STYLE_SPECS`` and the prompt builder straight
out of ``data_generation/make_safe_responses.py`` so the figure cannot drift from the code that
produced the data, and pulls the example refusals from the generated preview
CSV.

Usage
-----
    python analysis/plot_refusal_styles.py
    python analysis/plot_refusal_styles.py --behavior adv_training_1 --format png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_GENERATOR = REPO / "data_generation" / "make_safe_responses.py"
DEFAULT_PREVIEW = (REPO / "data/harmbench/optimizer_targets/extra_targets"
                   / "adv_training_targets_chosen4_preview.csv")
DEFAULT_OUT = HERE / "results" / "figures"

ROMAN = ["(i)", "(ii)", "(iii)", "(iv)"]

TEMPLATE_BG = "#eef2f5"
TEMPLATE_EDGE = "#93a7b5"
SLOT_BG = "#ffe9c9"
SLOT_EDGE = "#d99b3f"
CARD_BG = "#fbfbf8"
CARD_EDGE = "#c2cdd4"
OUT_BG = "#eaf4ef"
OUT_EDGE = "#1f8a70"
INK = "#1a1a1a"
MUTED = "#5a6672"

RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# The shared scaffold, condensed from build_generation_prompt() for display.
SYSTEM_LINE = ("System: You are writing preferred safe assistant responses for safety DPO "
               "training. Follow the requested structure exactly.")
SHARED_RULES = [
    "The first sentence must clearly refuse the request.",
    "Use at most 2 sentences, and follow the style's exact sentence count.",
    "Keep the response between 8 and 70 words.",
    "No harmful, illegal, or procedural details; no code, steps, links, or roleplay.",
    "Keep this style clearly distinct from the other styles.",
]


def load_style_specs(generator_path: Path) -> list[dict]:
    """Import STYLE_SPECS from the generation script itself."""
    spec = importlib.util.spec_from_file_location("_chosen4_generator", generator_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {generator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(module.STYLE_SPECS)


def load_examples(preview_path: Path, behavior_id: str) -> tuple[str, list[str]]:
    with preview_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["BehaviorID"] == behavior_id:
                return row["Behavior"], [row[f"style_{i}"] for i in range(1, 5)]
    raise SystemExit(f"behavior {behavior_id} not found in {preview_path}")


FIG_W = 7.16          # IEEE two-column text width, in inches
CHAR_W = 0.525        # mean glyph advance of DejaVu Serif, in units of font size
PAD_X = 0.009         # horizontal padding inside a box, as a figure fraction


def rounded(ax, x, y, w, h, facecolor, edgecolor, lw=0.9, zorder=1, aspect=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
        mutation_aspect=aspect,
    ))


def wrap(text: str, box_w: float, fontsize: float) -> list[str]:
    """Wrap to the character count that actually fits inside ``box_w``."""
    usable_pt = (box_w - 2 * PAD_X) * FIG_W * 72.0
    return textwrap.wrap(text, width=max(8, int(usable_pt / (CHAR_W * fontsize))))


def line_h(fontsize: float, fig_h: float, spacing: float = 1.33) -> float:
    """Height of one text line, as a fraction of figure height."""
    return fontsize * spacing / 72.0 / fig_h


def build_figure(styles: list[dict], behavior: str, examples: list[str], out_path: Path) -> None:
    n = len(styles)
    gap = 0.010
    card_w = (1.0 - gap * (n - 1)) / n

    fs_head, fs_body, fs_out = 7.6, 6.3, 6.4

    # Lay the text out first so the figure can be sized to fit it exactly.
    instructions = [wrap(s["instruction"], card_w, fs_body) for s in styles]
    outputs = [wrap(e, card_w, fs_out) for e in examples]
    rules = "  ".join(f"{i}. {r}" for i, r in enumerate(SHARED_RULES, start=1))
    system_lines = wrap(SYSTEM_LINE, 1.0, 6.8)
    rule_lines = wrap("Hard requirements:  " + rules, 1.0, 6.5)

    # Vertical budget, in inches, then converted once the height is known.
    inch = lambda k, fs, sp=1.33: k * fs * sp / 72.0
    tpl_in = 0.16 + inch(len(system_lines), 6.8) + 0.30 + inch(len(rule_lines), 6.5) + 0.08
    card_in = (0.20 + inch(max(len(v) for v in instructions), fs_body) + 0.18
               + 0.13 + inch(max(len(v) for v in outputs), fs_out) + 0.10)
    fig_h = tpl_in + 0.26 + card_in + 0.22

    fig = plt.figure(figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    aspect = FIG_W / fig_h

    def put(x, y, lines, fontsize, **kwargs):
        ax.text(x, y, "\n".join(lines), fontsize=fontsize, va="top",
                linespacing=1.33, **kwargs)
        return y - len(lines) * line_h(fontsize, fig_h)

    # ---- shared template block -------------------------------------------
    top = 1.0 - 0.06 / fig_h
    tpl_h = tpl_in / fig_h
    rounded(ax, 0.0, top - tpl_h, 1.0, tpl_h, TEMPLATE_BG, TEMPLATE_EDGE, aspect=aspect)

    y = top - 0.10 / fig_h
    y = put(PAD_X, y, ["Shared prompt template (identical for all four styles)"],
            8.0, fontweight="bold", color=INK)
    y = put(PAD_X, y - 0.02 / fig_h, system_lines, 6.8, color=MUTED)

    slot_h = 0.22 / fig_h
    slot_y = y - 0.04 / fig_h - slot_h
    rounded(ax, PAD_X, slot_y, 1.0 - 2 * PAD_X, slot_h, SLOT_BG, SLOT_EDGE,
            lw=1.0, zorder=2, aspect=aspect)
    ax.text(0.5, slot_y + slot_h / 2,
            r"Style goal:  $\langle$ style-specific instruction $-$ the only part that varies $\rangle$",
            fontsize=7.4, fontweight="bold", color="#7a4c07", ha="center", va="center", zorder=3)

    put(PAD_X, slot_y - 0.05 / fig_h, rule_lines, 6.5, color=MUTED)

    # ---- arrow from the slot down to the four instantiations -------------
    card_top = top - tpl_h - 0.26 / fig_h
    ax.add_patch(FancyArrowPatch((0.5, top - tpl_h - 0.04 / fig_h),
                                 (0.5, card_top + 0.03 / fig_h),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=SLOT_EDGE, linewidth=1.2))

    # ---- one card per style ----------------------------------------------
    card_h = card_in / fig_h
    out_h = (0.13 + inch(max(len(v) for v in outputs), fs_out) + 0.10) / fig_h

    for idx, (style, instruction, output) in enumerate(zip(styles, instructions, outputs)):
        x = idx * (card_w + gap)
        rounded(ax, x, card_top - card_h, card_w, card_h, CARD_BG, CARD_EDGE, aspect=aspect)

        y = card_top - 0.14 / fig_h
        y = put(x + PAD_X, y, [f"{ROMAN[idx]} {style['style_name']}"],
                fs_head, fontweight="bold", color=INK)
        put(x + PAD_X, y - 0.03 / fig_h, instruction, fs_body, color=MUTED)

        out_y = card_top - card_h + 0.07 / fig_h
        rounded(ax, x + 0.005, out_y, card_w - 0.010, out_h, OUT_BG, OUT_EDGE,
                lw=0.8, zorder=2, aspect=aspect)
        yy = put(x + PAD_X + 0.005, out_y + out_h - 0.05 / fig_h, ["Generated response"],
                 6.1, style="italic", color=OUT_EDGE, zorder=3)
        put(x + PAD_X + 0.005, yy - 0.01 / fig_h, output, fs_out, color=INK, zorder=3)

    ax.text(0.5, 0.06 / fig_h, f"Harmful request:  “{behavior}”",
            fontsize=6.7, style="italic", color=MUTED, ha="center", va="center")

    fig.savefig(out_path, dpi=400)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--behavior", default="adv_training_18",
                        help="behavior whose four refusals are shown as the example")
    parser.add_argument("--format", default="both", choices=["both", "pdf", "png"])
    args = parser.parse_args()

    styles = load_style_specs(args.generator)
    behavior, examples = load_examples(args.preview, args.behavior)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffixes = ["pdf", "png"] if args.format == "both" else [args.format]
    with plt.rc_context(RC):
        for suffix in suffixes:
            build_figure(styles, behavior, examples,
                         args.out_dir / f"refusal_styles.{suffix}")

    print(f"\nbehavior: {args.behavior}")
    for roman, style, example in zip(ROMAN, styles, examples):
        print(f"  {roman:5s} {style['style_name']:22s} {len(example.split()):2d} words")


if __name__ == "__main__":
    main()
