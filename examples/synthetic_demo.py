"""Exercise the pipeline on artificial data; no trained physics claims."""

import sys
from pathlib import Path

# Also allow running from a checkout before an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import cygnet


def run(output="tmp/figures/synthetic_demo.png"):
    """Save a labeled pipeline illustration and return its path."""
    torch.manual_seed(7)
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[:128, :128]
    signal = (abs(yy - 0.6 * xx - 20) < 2) & (xx > 16) & (xx < 112)
    image = np.clip(
        205 + rng.normal(0, 2, signal.shape) + 12 * signal, 0, 65535
    ).astype(np.uint16)
    pooled = cygnet.input_transform_builder(4)(image)
    # Match uint16 PNG -> float scaling used by CygnoSet.
    batch = pooled.float().unsqueeze(0) / 65535
    model = cygnet.SkipNet(base=2).eval()
    with torch.inference_mode():
        probabilities = model(batch).sigmoid()[0, 0]
    # Arbitrary demonstration threshold: deliberately separate from thesis tuning.
    mask = probabilities > 0.5
    full_mask = mask.repeat_interleave(4, 0).repeat_interleave(4, 1).numpy()
    panels = [image, signal, probabilities.numpy(), image * full_mask]
    titles = [
        "Synthetic input",
        "Known artificial track",
        "Untrained probabilities",
        "Selected pixels (demo only)",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Pipeline mechanics · randomly initialized model · no physics performance claim"
    )
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Synthetic pipeline completed: {output}")
    return output


if __name__ == "__main__":
    run()
