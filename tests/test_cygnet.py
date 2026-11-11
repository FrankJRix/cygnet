"""Scientific building-block checks on synthetic data, independent of detector files."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import uproot

import cygnet

torch.set_num_threads(2)


class TransformsAndModels(unittest.TestCase):
    def test_pooling_and_dilation(self):
        image = np.arange(64, dtype=np.uint16).reshape(8, 8)
        pooled = cygnet.input_transform_builder(4)(image)
        expected = np.round(image.reshape(2, 4, 2, 4).mean(axis=(1, 3)))
        np.testing.assert_array_equal(pooled.squeeze().numpy(), expected)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[3, 3] = 255
        target = cygnet.target_transform_builder(4)(mask)
        self.assertEqual(target[0, 0, 0], 255)
        self.assertEqual(target.count_nonzero(), 1)
        inflated = cygnet.inflate_transform_builder(2)(mask)
        self.assertEqual(inflated.count_nonzero(), 25)
        self.assertTrue((inflated[0, 1:6, 1:6] == 255).all())

    def test_forward_gradients_and_checkpoint_roundtrip(self):
        torch.manual_seed(4)
        for cls in (cygnet.Net, cygnet.SkipNet):
            model = cls(2)
            # Non-square, odd sizes exercise unpool output_size restoration.
            image = torch.randn(2, 1, 33, 49)
            target = torch.randint(0, 2, image.shape).float()
            logits = model(image)
            self.assertEqual(logits.shape, target.shape)
            loss = cygnet.binary_focal_loss(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grad = model.dc1.seq[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum(), 0)
            copy = cls(2)
            copy.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            copy.eval()
            with torch.inference_mode():
                torch.testing.assert_close(model(image), copy(image))

    def test_balanced_focal_loss_and_dice(self):
        logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0, 1.0])
        # At gamma=0 the focal term is exactly class-weighted BCE.
        expected = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * torch.where(target == 1, 0.25, 0.75)
        ).mean()
        torch.testing.assert_close(
            cygnet.binary_focal_loss(logits, target, gamma=0), expected
        )
        for mask in (torch.zeros(2, 8, 8), torch.ones(2, 8, 8)):
            self.assertEqual(cygnet.dice_coeff(mask, mask).item(), 1.0)

    def test_timer_contract(self):
        cygnet.start_time = None
        with self.assertRaises(RuntimeError):
            cygnet.end_timer_and_print("unstarted")
        with contextlib.redirect_stdout(io.StringIO()):
            cygnet.start_timer()
            cygnet.end_timer_and_print("CPU compatible")


class FileDatasets(unittest.TestCase):
    def test_png_pairing_scaling_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "mask").mkdir()
            for idx in (17, 2):
                iio.imwrite(
                    root / "input" / f"i_{idx}.png",
                    np.full((8, 8), idx, dtype=np.uint16),
                )
                iio.imwrite(
                    root / "mask" / f"m_{idx}.png", np.full((8, 8), 255, dtype=np.uint8)
                )
            (root / "input" / "notes.txt").write_text("ignored")
            data = cygnet.CygnoSet(root / "input", root / "mask")
            self.assertEqual(data.event_ids, [2, 17])
            image, mask = data[0]
            self.assertEqual(image.shape, (1, 8, 8))
            torch.testing.assert_close(image, torch.full_like(image, 2 / 65535))
            self.assertTrue((mask == 1).all())
            with self.assertRaises(IndexError):
                data[len(data)]
            (root / "mask" / "m_17.png").unlink()
            with self.assertRaises(ValueError):
                cygnet.CygnoSet(root / "input", root / "mask")

    def test_root_orientation_and_global_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for idx in range(2):
                image = np.arange(12).reshape(3, 4) + idx * 100
                with uproot.recreate(root / f"events_{idx}.root") as file:
                    file["event_info"] = {
                        "eventnumber": np.array([0]),
                        "redpix_ix": np.array([[1]]),
                        "redpix_iy": np.array([[2]]),
                        "redpix_iz": np.array([[19]]),
                    }
                    file["pic_run1_ev0"] = (image, np.arange(4), np.arange(5))
            (root / "notes.txt").write_text("ignored")
            manager = cygnet.RootManager(root)
            self.assertEqual(manager.total_events, 2)
            np.testing.assert_array_equal(
                manager.get_noisy(1), (np.arange(12).reshape(3, 4) + 100).T
            )
            clean = manager.get_clean(0)
            self.assertEqual(clean.shape, (4, 3))
            self.assertEqual(clean[2, 1], 255)
            self.assertEqual(np.count_nonzero(clean), 1)
            for idx in (-1, 2):
                with self.assertRaises(IndexError):
                    manager.get_noisy(idx)
            manager.ifc.file.close()


if __name__ == "__main__":
    unittest.main()
