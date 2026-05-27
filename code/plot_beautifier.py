"""
plot_beautifier.py
==================
Combines per-model plots into publication-ready composite figures.

Outputs (saved to FIG_ROOT/combined/):
  combined_sel_weight_gap.png         — 2×2 grid, one model per cell
  combined_alignment_error_dist.png   — 1×2: (GPT+LLAMA) | (MISTRAL+QWEN)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"
OUT_DIR  = os.path.join(FIG_ROOT, "combined")
MODELS   = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt": "GPT", "llama": "LLaMA", "mistral": "Mistral", "qwen": "Qwen"}

os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return mpimg.imread(path)


# ─── 1. sel_weight_gap — 2×2 grid ─────────────────────────────────────────────

def make_sel_weight_gap():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, model in zip(axes.flat, MODELS):
        path = os.path.join(FIG_ROOT, model, "selection_analysis", "sel_weight_gap.png")
        img  = load(path)
        ax.imshow(img)
        ax.set_title(MODEL_LABELS[model], fontsize=14, fontweight="bold", pad=6)
        ax.axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                        wspace=0.04, hspace=0.08)
    out = os.path.join(OUT_DIR, "combined_sel_weight_gap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─── 2. alignment_error_distribution — 1×2 (GPT+LLAMA | MISTRAL+QWEN) ─────────

def stack_vertical(model_a, model_b):
    """Stack two images vertically, matching widths."""
    path_a = os.path.join(FIG_ROOT, model_a, "alignment_analysis",
                          "alignment_error_distribution.png")
    path_b = os.path.join(FIG_ROOT, model_b, "alignment_analysis",
                          "alignment_error_distribution.png")
    img_a = load(path_a)
    img_b = load(path_b)

    # Normalise to [0,1] float if uint8
    if img_a.dtype == np.uint8:
        img_a = img_a.astype(np.float32) / 255.0
    if img_b.dtype == np.uint8:
        img_b = img_b.astype(np.float32) / 255.0

    # Match widths by cropping to the narrower width
    w = min(img_a.shape[1], img_b.shape[1])
    img_a = img_a[:, :w]
    img_b = img_b[:, :w]

    return np.concatenate([img_a, img_b], axis=0)


def save_side_by_side(model_a, model_b, fname):
    """Save two alignment_error_distribution images side by side (1 row, 2 cols)."""
    path_a = os.path.join(FIG_ROOT, model_a, "alignment_analysis",
                          "alignment_error_distribution.png")
    path_b = os.path.join(FIG_ROOT, model_b, "alignment_analysis",
                          "alignment_error_distribution.png")
    img_a = load(path_a)
    img_b = load(path_b)

    # Preserve aspect ratio: base figure width on combined pixel width
    h_px = max(img_a.shape[0], img_b.shape[0])
    w_px = img_a.shape[1] + img_b.shape[1]
    fig_w = 18.0
    fig_h = fig_w * (h_px / w_px)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    for ax, img, model in [(axes[0], img_a, MODEL_LABELS[model_a]),
                           (axes[1], img_b, MODEL_LABELS[model_b])]:
        ax.imshow(img)
        ax.set_title(model, fontsize=14, fontweight="bold", pad=6)
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                        wspace=0.02, hspace=0.0)
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def make_alignment_error_dist():
    save_side_by_side("gpt",     "llama",  "combined_alignment_error_dist_gpt_llama.png")
    save_side_by_side("mistral", "qwen",   "combined_alignment_error_dist_mistral_qwen.png")


if __name__ == "__main__":
    make_sel_weight_gap()
    make_alignment_error_dist()
    print("Done.")
