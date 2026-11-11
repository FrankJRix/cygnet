# Setup and external data

Use Python 3.10 or newer and install from the repository root with
`python -m pip install -e ".[notebooks]"`. A compatible PyTorch/torchvision pair
is required; choose the CUDA wheels appropriate for your hardware if needed.
`requirements-tested.txt` records the core versions used for CPU checks rather
than an exact reconstruction of the original training environment.

## PNG layout

```text
image_dataset/
  train/input/i_0.png
  train/mask/m_0.png
  train/mask_exp4/m4_0.png
  vali/input/i_0.png
  vali/mask/m_0.png
  vali/mask_exp4/m4_0.png
  test/input/i_0.png
  test/mask/m_0.png
  test/mask_exp4/m4_0.png
```

Each split contains many such pairs. IDs must match within input/target folders;
they need not be contiguous. Binary target PNGs use uint8 0/255. Inputs are
uint16 detector intensities. `CygnoSet` defaults to torchvision dtype scaling;
for uint16 input this divides by 65535, whereas uint8 targets divide by 255.
Keep training and inference normalization consistent. Custom transforms can be
supplied explicitly. Tensors have shape C×H×W; batches have shape B×C×H×W.

Set `CYGNET_DATA_DIR` to your image_dataset directory and
`CYGNET_CHECKPOINT_DIR` to a checkpoint directory before opening research
notebooks. Use `CYGNET_ROOT_DIR` for simulated ROOT files,
`CYGNET_MIDAS_FILE` for a real detector file and `CYGNET_FIGURES_DIR` for figures.
Some exploratory cells retain bespoke paths; see the notebook guide.

## ROOT and MIDAS

`RootInterface` expects an `event_info` tree with `eventnumber`, `redpix_ix`,
`redpix_iy`, `redpix_iz`, and histograms named `pic_run1_evN`. Its images are
transposed on reading. Files with other run naming need the historical extraction
notebook or an explicit adapter. `RootManager` scans only `.root` files.

MIDAS is optional: `open_mid` imports it only when called. Install the Python
package supplied by your DAQ MIDAS distribution (see the
[MIDAS project](https://daq00.triumf.ca/MidasWiki/index.php/Main_Page)). Do not
substitute an unrelated PyPI package with the same name. Some historical notebooks
also need experiment-specific `cygno`, PyROOT or visualization packages.

## Checkpoints

The model and registered layer names are unchanged. Historical checkpoints include
pickled model and loss objects alongside state dictionaries. Load only trusted
checkpoints; those payloads may require `weights_only=False`. Prefer newly exported
plain `state_dict` files for sharing. Instantiate `SkipNet(16)` and load matching
weights, use `eval()` and `torch.inference_mode()`, and apply sigmoid to logits.
The synthetic demo never loads external weights.

The repository does not include data or trained models, and the cleanup does not
establish redistribution rights for collaboration data or thesis illustrations.

The [checkpoint guide](checkpoint.md) provides a restricted loader for the supplied
historical payload and an optional plain-weight export. The weights are external.
