"""CYGNO image segmentation: detector I/O, preprocessing and PyTorch models.

Images use (height, width); tensors use (channels, height, width), with an
additional leading batch dimension for networks. Models return raw logits.
The historical class names and layer names are retained for saved checkpoints.
See docs/research.md for the thesis configuration and scientific conventions.
"""

import datetime
import gc
import os
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import uproot
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import tv_tensors

picmin = 198
picmax = 220

figures_path = str(Path(os.environ.get("CYGNET_FIGURES_DIR", "tmp/figures"))) + os.sep

# Detector I/O


class RootInterface:
    """Read simulated ROOT event_info branches and pic_run1_evN histograms.

    Returned images are transposed to the detector display convention. Binary
    targets contain 0/255; is_binary=False preserves uint16 signal counts."""

    def __init__(self, path, is_binary=True):
        self.load_file(path)
        self.is_binary = is_binary

    def load_file(self, path):
        self.file = uproot.open(path)
        tree = self.file["event_info"]
        self.evnum = len(tree["eventnumber"].arrays(library="np")["eventnumber"])
        self.ixs = tree["redpix_ix"].arrays(library="np")["redpix_ix"]
        self.iys = tree["redpix_iy"].arrays(library="np")["redpix_iy"]
        self.izs = tree["redpix_iz"].arrays(library="np")["redpix_iz"]
        self.p_shape = self.file[f"pic_run1_ev{0}"].to_numpy()[0].shape

    def get_num_of_events(self):
        return self.evnum

    def get_noisy_image(self, idx):
        return self.file[f"pic_run1_ev{idx}"].to_numpy()[0].T

    def get_clean_image(self, idx):
        ix_event = self.ixs[idx]
        iy_event = self.iys[idx]
        iz_event = self.izs[idx]
        image = np.zeros(self.p_shape, dtype=np.uint8 if self.is_binary else np.uint16)
        image[ix_event, iy_event] = 255 if self.is_binary else iz_event

        gc.collect()

        return image.T

    def plot_noisy_image(self, idx, num_title=-1):
        if num_title == -1:
            num_title = idx
        img = self.get_noisy_image(idx)
        plt.figure(figsize=(8, 8))
        plt.title(f"EVENT #{num_title}")
        plt.imshow(img, vmin=picmin, vmax=picmax)
        plt.show()

    def plot_clean_image(self, idx, num_title=-1):
        if num_title == -1:
            num_title = idx
        img = self.get_clean_image(idx)
        plt.figure(figsize=(8, 8))
        plt.title(f"redpix of EVENT #{num_title}")
        plt.imshow(img)
        plt.show()

    def plot_event(self, idx, num_title=-1):
        if num_title == -1:
            num_title = idx
        noisy = self.get_noisy_image(idx)
        clean = self.get_clean_image(idx)

        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(noisy, vmin=picmin, vmax=picmax)
        axs[0].set_title(f"EVENT #{num_title}")
        axs[1].imshow(clean)
        axs[1].set_title(f"redpix of EVENT #{num_title}")


class RootManager:
    """Index events across sorted ROOT files in one directory.

    Keeps one active file; use a separate instance per data-loader worker."""

    def __init__(self, dir):
        self.data_dir = dir
        self.files = sorted(p.name for p in Path(dir).glob("*.root") if p.is_file())
        if not self.files:
            raise ValueError(f"No ROOT files found in {dir}")

        self.events_per_file_dict = {}
        self.rootfile_starts_dict = {}
        sum = 0
        for file in self.files:
            path = os.path.join(self.data_dir, file)
            ifc = RootInterface(path)
            tmp = ifc.get_num_of_events()
            ifc.file.close()

            self.rootfile_starts_dict[file] = sum
            self.events_per_file_dict[file] = tmp
            sum += tmp
        self.total_events = sum
        self.rootfile_starts_dict["end"] = sum

        self.load_file(self.files[0])

    def load_file(self, file):
        path = os.path.join(self.data_dir, file)
        if hasattr(self, "ifc"):
            self.ifc.file.close()
        self.ifc = RootInterface(path)
        self.current_file = file
        gc.collect()

    def update_file_pointer(self, idx):
        if idx >= self.total_events or idx < 0:
            raise IndexError(f"Event index {idx} outside [0, {self.total_events})")

        file = ""
        for key, val in self.rootfile_starts_dict.items():
            if idx >= val:
                file = key
            else:
                break

        if file == "":
            print("SOMETHING WENT WRONG")
            return
        elif file != self.current_file:
            self.load_file(file)

        # self.ifc.plot_event(idx - offset, idx)

    def get_noisy(self, idx):
        self.update_file_pointer(idx)
        offset = self.rootfile_starts_dict[self.current_file]
        return self.ifc.get_noisy_image(idx - offset)

    def get_clean(self, idx):
        self.update_file_pointer(idx)
        offset = self.rootfile_starts_dict[self.current_file]
        return self.ifc.get_clean_image(idx - offset)

    def get_pair(self, idx):
        return self.get_noisy(idx), self.get_clean(idx)


def print_through(x):
    print("\n")
    print(x)
    print("\n")
    return x


# Preprocessing and datasets


class AvgTransform(nn.Module):
    """Downsample spatial dimensions using non-overlapping average pooling."""

    def __init__(self, k_size):
        super().__init__()
        self.pooling = torch.nn.AvgPool2d(kernel_size=k_size)

    def forward(self, x):
        return self.pooling(x)


class MaxTransform(nn.Module):
    """Downsample a target mask while retaining any foreground pixel."""

    def __init__(self, k_size):
        super().__init__()
        self.pooling = torch.nn.MaxPool2d(kernel_size=k_size)

    def forward(self, x):
        return self.pooling(x)


class InflateTransform(nn.Module):
    """Dilate a mask with a square (2 * radius + 1) kernel at stride one."""

    def __init__(self, radius):
        super().__init__()

        k_size = 2 * radius + 1
        self.pooling = torch.nn.MaxPool2d(kernel_size=k_size, padding=radius, stride=1)

    def forward(self, x):
        return self.pooling(x)


def input_transform_builder(pool_k_size):
    """Average pool, round, and retain uint16 detector intensities (no scaling)."""
    downscaler = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False),
            AvgTransform(pool_k_size),
            torch.round,
            transforms.ToDtype(torch.uint16, scale=False),
        ]
    )
    return downscaler


def target_transform_builder(pool_k_size):
    """Max pool and retain uint8 target labels (0/255 for binary masks)."""
    downscaler = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False),
            MaxTransform(pool_k_size),
            transforms.ToDtype(torch.uint8, scale=False),
        ]
    )
    return downscaler


def inflate_transform_builder(radius):
    """Build square mask dilation preserving uint8 target values."""
    downscaler = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False),
            InflateTransform(radius),
            transforms.ToDtype(torch.uint8, scale=False),
        ]
    )
    return downscaler


class RootDataset(Dataset):
    """Yield noisy image/target pairs from ROOT, optionally pooled by pool_k_size."""

    def __init__(self, dir, pool_k_size, raw=False):
        self.rmngr = RootManager(dir)
        self.transform_input = input_transform_builder(pool_k_size)
        self.transform_target = target_transform_builder(pool_k_size)
        self.raw = raw

    def __len__(self):
        return self.rmngr.total_events

    def __getitem__(self, idx):
        img, target = self.rmngr.get_pair(idx)
        if not self.raw:
            img = self.transform_input(img)
            target = self.transform_target(target)

        return img, target


itrtr = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)
trtr = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)


class CygnoSet(Dataset):
    """Load paired PNGs named prefix_INDEX.png from input and target directories.

    Numeric IDs must match, but need not be contiguous. Default transforms
    scale uint8 masks to [0, 1]; uint16 images retain torchvision dtype scaling."""

    def __init__(
        self, input_dir, target_dir, transform_input=itrtr, transform_target=trtr
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir

        def indexed_pngs(directory):
            result = {}
            for path in Path(directory).glob("*.png"):
                try:
                    idx = int(path.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Expected prefix_INDEX.png: {path.name}") from exc
                if idx in result:
                    raise ValueError(f"Duplicate image ID {idx} in {directory}")
                result[idx] = path
            return result

        inputs = indexed_pngs(input_dir)
        targets = indexed_pngs(target_dir)
        if not inputs or inputs.keys() != targets.keys():
            raise ValueError(
                "Input and target PNGs must have identical, nonempty ID sets"
            )
        self.event_ids = sorted(inputs)
        self.input_images = [inputs[idx] for idx in self.event_ids]
        self.target_images = [targets[idx] for idx in self.event_ids]
        self.input_prefix = self.input_images[0].stem.rsplit("_", 1)[0]
        self.mask_prefix = self.target_images[0].stem.rsplit("_", 1)[0]

        self.transform_input = transform_input
        self.transform_target = transform_target

        self.stats = {}

    def __len__(self):
        return len(self.input_images)

    def get_raw_item(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Dataset index {idx} outside [0, {len(self)})")
        input = iio.imread(self.input_images[idx])
        target = iio.imread(self.target_images[idx])
        if input.shape != target.shape:
            raise ValueError(
                f"Image/target shape mismatch for event {self.event_ids[idx]}"
            )

        return input, target

    def __getitem__(self, idx):
        input, target = self.get_raw_item(idx)
        target = tv_tensors.Mask(target).unsqueeze(0)

        if self.transform_input:
            input = self.transform_input(input)
        if self.transform_target:
            target = self.transform_target(target)

        return input, target

    def print_stats(self):
        print("=== dataset stats: ===")
        for key, item in self.stats.items():
            print(f"{key}: {item}")
        print()

    def compute_stats(self, raw=True):
        """Print and store per-image mean intensity averaged over the dataset."""

        if raw:
            getter = self.get_raw_item
        else:
            getter = self.__getitem__

        i_means = []

        m_means = []

        for ii in range(len(self)):
            image, mask = getter(ii)
            i_means.append(image.mean())
            m_means.append(mask.mean())

        i_mean = np.array(i_means).mean()
        m_mean = np.array(m_means).mean()

        self.stats["input_mean"] = i_mean
        self.stats["mask_mean"] = m_mean

        self.print_stats()


################### Utilities


def debug_plot(noisy, mask, alpha=0.1, vmin=picmin, vmax=picmax):
    plt.figure(figsize=(12, 12))
    plt.title("MASKED")
    plt.imshow(noisy, vmin=vmin, vmax=vmax)
    plt.imshow(mask, alpha=alpha)
    plt.tight_layout()
    plt.show()


def quickplot(img, title="", vmax=picmax, vmin=picmin):
    plt.figure()
    plt.imshow(img, vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def open_mid(path):
    """Open a MIDAS file; requires the optional DAQ MIDAS Python package."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        from midas.file_reader import MidasFile
    except ImportError as exc:
        raise ImportError(
            "MIDAS support requires the DAQ MIDAS Python package; see docs/setup.md"
        ) from exc
    return MidasFile(path)


start_time = None


def start_timer():
    """Start a wall-clock timer, synchronizing and resetting CUDA if available."""
    global start_time
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()


def end_timer_and_print(local_msg):
    """Print elapsed seconds and, on CUDA, peak tensor memory in GiB."""
    if start_time is None:
        raise RuntimeError("Call start_timer() before ending the timer")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("\n" + local_msg)
    print(f"Total execution time = {(end_time - start_time):.3f} sec")
    if torch.cuda.is_available():
        print(
            f"Max memory used by tensors = {torch.cuda.max_memory_allocated() / 1024**3:.3f} Gbytes"
        )


class SaveBestModel:
    """Save the best validation checkpoint using the historical payload format."""

    def __init__(self, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.t = datetime.datetime.now().strftime("%Y_%b_%d__%H:%M")

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, base):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")

            Path("saved_models").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f"saved_models/{self.t}_best_model_base{base}_res_drop.pt",
            )


# Network
dout = 0.0


class DoubleDown2(nn.Module):
    """Two convolution units followed by indexed max pooling."""

    def __init__(self, chin, chout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chout, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
        )
        self.mp = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape


class DoubleDown3(nn.Module):
    """Three convolution units followed by indexed max pooling."""

    def __init__(self, chin, chout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chout, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chout, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
        )
        self.mp = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        pool_shape = y.shape
        y, indices = self.mp(y)
        return y, indices, pool_shape


class DoubleUp2(nn.Module):
    """Indexed max unpooling followed by two convolution units."""

    def __init__(self, chin, chout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chin, 3, padding=1, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chin, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
        )
        self.mup = nn.MaxUnpool2d(2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y


class DoubleUp3(nn.Module):
    """Indexed max unpooling followed by three convolution units."""

    def __init__(self, chin, chout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chin, 3, padding=1, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chin, chin, 3, padding=1, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chin, chout, 3, padding=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Dropout(dout),
        )
        self.mup = nn.MaxUnpool2d(2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y


class DoubleUp2Out(nn.Module):
    """Unpool and decode to raw logits without a final activation."""

    def __init__(self, chin, chout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chin, 3, padding=1, bias=False),
            nn.BatchNorm2d(chin),
            nn.ReLU(),
            nn.Dropout(dout),
            nn.Conv2d(chin, chout, 3, padding=1, bias=False),
            # nn.BatchNorm2d(chout),
            # nn.ReLU(),
        )
        self.mup = nn.MaxUnpool2d(2)

    def forward(self, x, indices, output_size):
        y = self.mup(x, indices, output_size=output_size)
        y = self.seq(y)
        return y


class Net(nn.Module):
    """Three active encoder/decoder levels without additive skip connections.

    Deeper registered layers remain for historical checkpoint compatibility."""

    def __init__(self, base):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)

        self.dc1 = DoubleDown2(1, base)
        self.dc2 = DoubleDown2(base, base * 2)
        self.dc3 = DoubleDown3(base * 2, base * 4)
        self.dc4 = DoubleDown3(base * 4, base * 8)
        self.dc5 = DoubleDown3(base * 8, base * 8)

        self.uc5 = DoubleUp3(base * 8, base * 8)
        self.uc4 = DoubleUp3(base * 8, base * 4)
        self.uc3 = DoubleUp3(base * 4, base * 2)
        self.uc2 = DoubleUp2(base * 2, base)
        self.uc1 = DoubleUp2Out(base, 1)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)

        x, mp1_indices, shape1 = self.dc1(x)
        x, mp2_indices, shape2 = self.dc2(x)
        x, mp3_indices, shape3 = self.dc3(x)
        # x, mp4_indices, shape4 = self.dc4(x)
        # x, mp5_indices, shape5 = self.dc5(x)

        # x = self.uc5(x, mp5_indices, output_size=shape5)
        # x = self.uc4(x, mp4_indices, output_size=shape4)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = self.uc1(x, mp1_indices, output_size=shape1)

        return x


class SkipNet(nn.Module):
    """Thesis hybrid SegNet/UNet with four active levels and additive skips.

    base=16 gives active encoder widths 16, 32, 64, 128. The fifth registered
    level is inactive and retained for checkpoint compatibility."""

    def __init__(self, base):
        super().__init__()
        self.bn_input = nn.BatchNorm2d(1)

        self.dc1 = DoubleDown2(1, base)
        self.dc2 = DoubleDown2(base, base * 2)
        self.dc3 = DoubleDown3(base * 2, base * 4)
        self.dc4 = DoubleDown3(base * 4, base * 8)
        self.dc5 = DoubleDown3(base * 8, base * 8)

        self.uc5 = DoubleUp3(base * 8, base * 8)
        self.uc4 = DoubleUp3(base * 8, base * 4)
        self.uc3 = DoubleUp3(base * 4, base * 2)
        self.uc2 = DoubleUp2(base * 2, base)
        self.uc1 = DoubleUp2Out(base, 1)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)

        x1, mp1_indices, shape1 = self.dc1(x)
        x2, mp2_indices, shape2 = self.dc2(x1)
        x3, mp3_indices, shape3 = self.dc3(x2)
        x4, mp4_indices, shape4 = self.dc4(x3)
        # x, mp5_indices, shape5 = self.dc5(x)

        # x = self.uc5(x, mp5_indices, output_size=shape5)
        x = self.uc4(x4, mp4_indices, output_size=shape4)
        x = self.uc3(x3 + x, mp3_indices, output_size=shape3)
        x = self.uc2(x + x2, mp2_indices, output_size=shape2)
        x = self.uc1(x + x1, mp1_indices, output_size=shape1)

        return x


# Metrics and Losses


def focal_loss(pred, target, alpha=0.3, gamma=2.0):
    """Legacy focal variant with a constant alpha multiplier for both classes.

    Distinct from class-balanced binary_focal_loss used by the thesis recipe.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Class-balanced focal loss on logits; targets are binary floats in [0, 1]."""
    probs = torch.sigmoid(inputs)
    targets = targets.float()

    # Compute binary cross entropy
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    # Compute focal weight
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    bce_loss = alpha_t * bce_loss

    # Apply focal loss weighting
    loss = focal_weight * bce_loss

    return loss.mean()


def IoUMetric(pred, gt, softmax=False):
    """Legacy soft three-class IoU; gt contains integer labels 0, 1, 2.

    Do not apply this helper to single-channel binary network output.
    """
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    # end if

    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([(gt == i) for i in range(3)], dim=1)
    # print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)

    # Compute the mean over the batch dimension.
    return iou.mean()


class IoULoss(nn.Module):
    """Negative log of the legacy three-class soft IoU (not a binary metric)."""

    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax

    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax).log())


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    """Mean Dice overlap of same-shaped masks/probabilities; empty pairs score one."""
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor):
    """One minus Dice overlap after converting binary logits to probabilities."""
    return 1 - dice_coeff(torch.sigmoid(input), target, reduce_batch_first=False)
