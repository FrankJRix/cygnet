"""Inspect a thesis checkpoint and exercise it on artificial input, without data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import sigmoid_focal_loss

import cygnet


def load_model(path):
    """Load a state dict or the known historical payload with restricted globals.

    Only the project's own model classes and known PyTorch layers are permitted;
    arbitrary globals requested by a serialized file are never imported.
    """
    allowed = [
        cygnet.SkipNet,
        cygnet.DoubleDown2,
        cygnet.DoubleDown3,
        cygnet.DoubleUp2,
        cygnet.DoubleUp3,
        cygnet.DoubleUp2Out,
        torch.nn.Sequential,
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.Dropout,
        torch.nn.MaxPool2d,
        torch.nn.MaxUnpool2d,
        sigmoid_focal_loss,
    ]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected a state dictionary or a historical checkpoint dictionary"
        )
    state = payload.get("model_state_dict", payload)
    base = state["dc1.seq.0.weight"].shape[0]
    model = cygnet.SkipNet(base)
    model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains nonfinite parameters or buffers")
    return model.eval(), payload, base


def run(checkpoint, output, export_weights=None):
    """Check native-sized inference and save an explicitly artificial illustration."""
    torch.set_num_threads(2)
    model, payload, base = load_model(checkpoint)
    rng = np.random.default_rng(7)
    # Native camera geometry, but no claim to model the detector or its simulation.
    yy, xx = np.ogrid[:2304, :4096]
    track = (abs(yy - 0.35 * xx - 300) < 8) & (xx > 600) & (xx < 3500)
    image = np.clip(205 + rng.normal(0, 2, track.shape) + 12 * track, 0, 65535).astype(
        np.uint16
    )
    pooled = cygnet.input_transform_builder(4)(image)
    batch = cygnet.itrtr(pooled).unsqueeze(0)
    with torch.inference_mode():
        logits = model(batch)
        probabilities = logits.sigmoid()[0, 0]
    if logits.shape != batch.shape or not torch.isfinite(logits).all():
        raise ValueError("Inference failed shape or finiteness checks")
    threshold = 0.15
    mask = (
        (probabilities > threshold)
        .repeat_interleave(4, 0)
        .repeat_interleave(4, 1)
        .numpy()
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "model": "SkipNet",
        "base": base,
        "epoch": payload.get("epoch"),
        "checkpoint_keys": sorted(payload)
        if "model_state_dict" in payload
        else ["plain state_dict"],
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_load": "strict; all keys matched",
        "parameter_and_buffer_values": "all finite",
        "input_shape": list(batch.shape),
        "output_shape": list(logits.shape),
        "output_values": "all finite",
        "artificial_probability_range": [
            float(probabilities.min()),
            float(probabilities.max()),
        ],
        "artificial_selected_pixel_fraction": float(mask.mean()),
        "input_batchnorm_running_mean": model.bn_input.running_mean.tolist(),
        "threshold": threshold,
        "data": "Artificial line plus Gaussian noise; not detector data or validated simulation",
        "interpretation": "Technical inference check only; no physics performance metrics computed",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (image, "Artificial input", 198, 225),
        (probabilities.numpy(), "Checkpoint probabilities", 0, 1),
        (image * mask, "Selected pixels at threshold 0.15", 198, 225),
    ]
    for ax, (panel, title, low, high) in zip(axes, panels):
        ax.imshow(panel, vmin=low, vmax=high)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Trained checkpoint · artificial input · technical check only")
    fig.tight_layout()
    fig.savefig(output / "checkpoint_demo.png", dpi=150)
    plt.close(fig)
    if export_weights:
        destination = Path(export_weights)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/checkpoint_demo"))
    parser.add_argument(
        "--export-weights", type=Path, help="Optional plain state_dict output"
    )
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.export_weights)
