# CYGNET · Physics-aware image segmentation

**Keep the detector signal. Retain fewer pixels.**

CYGNET explores neural-network segmentation as an early data-reduction step for
CYGNO, a dark-matter experiment using an optically read out gas time projection
chamber. Its high-resolution camera images are dominated by instrumental noise;
the physically interesting tracks occupy a small fraction of each frame.

Developed by **Franco Risi** for the physics thesis *Metodi di Machine Learning
applicati all’esperimento CYGNO per la ricerca della materia oscura*
(Sapienza Università di Roma, academic year 2025–2026).

The objective is conservative region selection: preserve signal energy and track
structure while discarding background pixels before later reconstruction.

## Approach

1. Average-pool a 4096 × 2304 camera frame by four along each axis.
2. Segment the resulting 1024 × 576 image using a hybrid UNet/SegNet network.
3. Apply sigmoid and a probability threshold of 0.15.
4. Expand the mask to native resolution and select pixels from the original frame.

The network combines indexed max unpooling with additive skip connections.
Training uses focal loss to address class imbalance, and dilated targets to favor
signal recovery over tight boundaries. See [the research reference](docs/research.md)
for the dataset, architecture, training recipe, and scientific caveats.

[View the threshold tradeoff and failure-case guide](docs/results-gallery.md).

## Thesis results

These are **reported thesis results**, not measurements reproduced by this cleanup.
Evaluation used 900 simulated test events; timing used 200 frames on a Tesla T4.

| Quantity | Reported value |
| --- | ---: |
| Precision | 0.79 ± 0.12 |
| Recall | 0.99 ± 0.06 |
| IoU | 0.79 ± 0.12 |
| Dice | 0.88 ± 0.10 |
| Mean missed energy | (0.12 ± 0.11)% |
| Mean pixel retention | (0.38 ± 0.02)% |
| Network inference | (17.6 ± 0.2) ms/frame |
| Full pipeline | (47 ± 2) ms/frame |

EM and PR uncertainties are statistical uncertainties on the mean; the other
uncertainties are distribution standard deviations. Pixel retention measures
selected pixels, **not actual compressed file size**. Of 900 test frames, 893 had
zero missed energy; one very weak 1 keV low-gain event was missed entirely.
Real-data examples in the thesis are qualitative, without ground-truth validation.

## Try it without detector data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebooks]"
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
jupyter lab
```

The synthetic demonstration checks preprocessing, a small network, and mask
selection on CPU. Its randomly initialized model is **not a trained detector
model**, and its output makes no claim about physics performance.

Start with [the walkthrough notebook](examples/walkthrough.ipynb), then consult
[the notebook guide](docs/notebooks.md) for the original research workflow.
[Setup and data conventions](docs/setup.md) describe optional MIDAS support and
how to supply detector data and checkpoints. Core CPU versions used for
verification are recorded in `requirements-tested.txt`.

## Supplied checkpoint

The author-provided checkpoint was verified as `SkipNet(16)`, epoch 15. It loads
strictly and runs at the thesis input size. See the [checkpoint guide](docs/checkpoint.md)
for the reproducible loading example and its technical check report. Weights remain
external; the artificial-input diagnostic is not a physics validation.

## Repository map

| Location | Purpose |
| --- | --- |
| `cygnet.py` | ROOT/PNG datasets, transforms, models, losses and legacy utilities |
| `examples/` | Self-contained CPU demonstration and narrated walkthrough |
| `train.ipynb` | Historical training and pixel-level evaluation |
| `metrics.ipynb` | Threshold sweeps, signal retention and qualitative examples |
| `image_extractor.ipynb` | Historical ROOT-to-PNG dataset preparation |
| `docs/` | Scientific reference, setup, notebook map and maintenance notes |
| `tests/` | Synthetic checks of data, models, losses and preprocessing |

The root notebooks preserve their recorded outputs and experimental history.
They require external data and may reflect different stages of the thesis work.
No datasets, trained weights or thesis PDF are bundled here. Historical run exports
remain available in the checkout; they are not treated as a canonical results table.
