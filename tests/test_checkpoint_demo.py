"""Check supported checkpoint formats without distributing trained weights."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.ops import sigmoid_focal_loss

import cygnet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from checkpoint_demo import load_model


class CheckpointFormats(unittest.TestCase):
    def test_plain_and_historical_formats(self):
        torch.manual_seed(11)
        original = cygnet.SkipNet(2).eval()
        image = torch.randn(1, 1, 32, 48)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for payload in (
                original.state_dict(),
                {
                    "model": original,
                    "model_state_dict": original.state_dict(),
                    "loss": sigmoid_focal_loss,
                    "epoch": 15,
                },
            ):
                torch.save(payload, path)
                loaded, metadata, base = load_model(path)
                self.assertEqual(base, 2)
                self.assertFalse(loaded.training)
                with torch.inference_mode():
                    torch.testing.assert_close(
                        loaded(image), original(image), rtol=0, atol=0
                    )

    def test_nonfinite_weights_are_rejected(self):
        state = cygnet.SkipNet(2).state_dict()
        state["dc1.seq.0.weight"][0, 0, 0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_model(path)
