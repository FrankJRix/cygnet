"""Export pooled ROOT image/target pairs; does not create thesis dataset splits."""

import argparse
from pathlib import Path

import imageio.v3 as iio

import cygnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory of pic_run1 ROOT files")
    parser.add_argument(
        "destination", type=Path, help="Output directory (input/ and mask/)"
    )
    parser.add_argument("--pool", type=int, default=4)
    args = parser.parse_args()
    if args.pool < 1:
        parser.error("--pool must be positive")
    dataset = cygnet.RootDataset(args.source, args.pool)
    # Require a new destination to prevent accidentally mixing datasets.
    args.destination.mkdir(parents=True, exist_ok=False)
    for folder in ("input", "mask"):
        (args.destination / folder).mkdir()
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        iio.imwrite(
            args.destination / "input" / f"i_{idx}.png", image.squeeze(0).numpy()
        )
        iio.imwrite(args.destination / "mask" / f"m_{idx}.png", mask.squeeze(0).numpy())
    print(f"Exported {len(dataset)} pairs to {args.destination}")


if __name__ == "__main__":
    main()
