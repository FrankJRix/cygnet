# Cleanup and compatibility

The library remains a single importable module so notebooks and pickled model
paths stay compatible. Its sections separate detector I/O, transforms/datasets,
utilities, network blocks and losses. Public model names, forward computations,
layer names and historical checkpoint payloads are preserved.

Changes with observable behavior:

- PNG pairs are matched by numeric ID and checked for missing/duplicate IDs and
  shape mismatch. Non-contiguous IDs now work; unrelated non-PNG files are ignored.
- ROOT directory scans ignore non-ROOT entries, reject empty directories and raise
  `IndexError` on invalid global indices instead of continuing with a stale pointer.
- ROOT files opened during indexing and file switching are closed.
- MIDAS is loaded only when requested. Timing works on CPU as well as CUDA.
- Best-checkpoint saving creates its destination directory.
- Notebook configuration paths can be supplied by environment variables.
- `image_extractor.py` is now an explicit ROOT-to-PNG CLI instead of broken scratch
  code. It does not recreate the full stratified thesis split or dilated targets.

Historical output cells are preserved. Their results predate these changes.
Synthetic tests check tensor behavior and gradients, pooling/dilation, file pairing,
ROOT orientation/indexing and state-dictionary compatibility. They do not validate
physics performance. No retraining or Tesla T4 benchmark is performed.

## Verification performed

On the prepared CPU environment (PyTorch 2.7.1, torchvision 0.22.1):

- Six unittest cases passed, including forward/backward checks on both networks.
- Original pre-cleanup state dictionaries loaded strictly into both models, and
  evaluation outputs on identical synthetic tensors were bit-identical.
- The editable package built and installed with the existing dependencies.
- The synthetic script and walkthrough notebook executed successfully.
- All root notebook schemas validated and all historical output cells matched
  the pre-cleanup versions.
- Ruff lint/format checks and Git whitespace checks passed.

The GitHub Actions workflow is provided but has not yet run remotely. Historical
research notebooks were not rerun without the external datasets and checkpoints.
