"""Generate the images shown in README.md (docs/images/).

Uses the same OpenCV operations and default parameters as the C++ app:
cvtColor(BGR2GRAY) -> GaussianBlur(5x5) -> Canny(50, 150).

    python scripts/make_readme_images.py
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
W, H = 640, 400


def make_scene() -> np.ndarray:
    """Draw a simple landscape scene (BGR) with clear edges plus light noise."""
    img = np.zeros((H, W, 3), np.uint8)

    # Sky gradient (light blue at the horizon, deeper blue at the top).
    for y in range(H):
        t = y / H
        img[y, :] = (235 - 60 * (1 - t), 200 - 70 * (1 - t), 120 + 60 * t)

    cv2.circle(img, (520, 85), 45, (60, 210, 255), -1)  # sun

    mountains = np.array([[0, 300], [120, 150], [230, 260], [360, 120], [500, 270], [640, 180], [640, 400], [0, 400]])
    cv2.fillPoly(img, [mountains], (110, 120, 95))
    cv2.rectangle(img, (0, 310), (W, H), (70, 150, 80), -1)  # grass

    # House.
    cv2.rectangle(img, (150, 230), (300, 340), (90, 140, 200), -1)
    cv2.fillPoly(img, [np.array([[135, 232], [225, 170], [315, 232]])], (50, 50, 160))
    cv2.rectangle(img, (205, 280), (245, 340), (40, 70, 110), -1)  # door
    for x in (165, 260):
        cv2.rectangle(img, (x, 250), (x + 25, 275), (230, 230, 200), -1)  # windows

    # Tree.
    cv2.rectangle(img, (440, 260), (455, 340), (40, 70, 100), -1)
    cv2.circle(img, (447, 245), 35, (50, 120, 40), -1)

    noise = np.random.default_rng(42).normal(0, 6, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def process(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    return gray, edges


def save_pipeline_figure(bgr, gray, edges, path: Path) -> None:
    """Original -> Grayscale -> Edges, side by side with arrows and captions."""
    panels = [
        (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "1. Original", "3 channels (BGR)", None),
        (gray, "2. Grayscale", "cv::cvtColor  ·  1 channel", "gray"),
        (edges, "3. Canny edges", "blur 5x5  ·  Canny 50 / 150", "gray"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
    for ax, (img, title, sub, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel(sub, fontsize=11, color="#555")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbb")

    fig.tight_layout(w_pad=4)
    for a, b in ((axes[0], axes[1]), (axes[1], axes[2])):
        x0 = a.get_position().x1 + 0.004
        x1 = b.get_position().x0 - 0.004
        y = (a.get_position().y0 + a.get_position().y1) / 2
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=22, lw=2.5, color="#2563eb"))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_flow_diagram(path: Path) -> None:
    """Box-and-arrow diagram of the timed processing stages."""
    stages = [
        ("Input image", "png / jpg / bmp", "#e5e7eb"),
        ("1. Load", "cv::imread", "#dbeafe"),
        ("2. Grayscale", "cv::cvtColor", "#dbeafe"),
        ("3. Edges", "Blur + Canny", "#dbeafe"),
        ("4. Save", "cv::imwrite", "#dbeafe"),
        ("Output", "_gray.png\n_edges.png", "#dcfce7"),
    ]
    timed = {1, 2, 3, 4}
    fig, ax = plt.subplots(figsize=(15, 3.6), facecolor="white")
    ax.set_xlim(0, len(stages) * 2.5), ax.set_ylim(0, 3.6), ax.axis("off")

    bw, bh, y = 2.0, 1.3, 1.6
    for i, (title, sub, color) in enumerate(stages):
        x = i * 2.5 + 0.25
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    fc=color, ec="#1e3a8a" if i in timed else "#6b7280", lw=1.8))
        ax.text(x + bw / 2, y + bh * 0.66, title, ha="center", va="center", fontsize=12.5, fontweight="bold")
        ax.text(x + bw / 2, y + bh * 0.30, sub, ha="center", va="center", fontsize=10, color="#374151",
                family="monospace")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.5, y + bh / 2), xytext=(x + bw + 0.1, y + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#2563eb", mutation_scale=18))

    # Bracket under the timed stages.
    x_start, x_end = 1 * 2.5 + 0.25, 4 * 2.5 + 0.25 + bw
    ax.plot([x_start, x_start, x_end, x_end], [1.3, 1.1, 1.1, 1.3], color="#b45309", lw=1.8)
    ax.text((x_start + x_end) / 2, 0.7, "each stage timed separately with std::chrono::steady_clock",
            ha="center", va="center", fontsize=11.5, color="#b45309", style="italic")
    ax.text(len(stages) * 1.25, 3.35, "Image Processing Pipeline", ha="center", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bgr = make_scene()
    gray, edges = process(bgr)

    cv2.imwrite(str(OUT / "original.png"), bgr)
    cv2.imwrite(str(OUT / "grayscale.png"), gray)
    cv2.imwrite(str(OUT / "edges.png"), edges)
    save_pipeline_figure(bgr, gray, edges, OUT / "pipeline_result.png")
    save_flow_diagram(OUT / "pipeline_diagram.png")
    print(f"Wrote images to {OUT}")


if __name__ == "__main__":
    main()
