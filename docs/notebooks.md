# Notebook guide

Start with `examples/walkthrough.ipynb` for a runnable demonstration without data.
Original research notebooks remain at the root so their imports and links stay usable.
Their output cells have been preserved; changing a configuration cell does not update historical outputs.

## Research workflow

1. `image_extractor.ipynb`: prepare simulation images and masks.
2. `train.ipynb`: fit a model and select by validation loss.
3. `metrics.ipynb`: examine physics retention and threshold sweeps.
4. `midas_test.ipynb`: qualitative real-data application.

Set environment paths as described in [setup.md](setup.md). Inspect all cells before running: extraction writes files, training saves models, and some exploratory notebooks access external data. Exact per-event paths in scratch cells still need local configuration. Historical training/evaluation choices are documented in [research.md](research.md).

## Inventory

| Notebook | Purpose |
| --- | --- |
| [image_extractor.ipynb](../image_extractor.ipynb) | Prepare simulated ROOT events as pooled PNG images and original/dilated target masks. Historical extraction logic includes manual split counters; use a new destination and audit event IDs before regenerating data. |
| [metric_test_extractor.ipynb](../metric_test_extractor.ipynb) | Prepare native-resolution images and signal-count maps for physics evaluation. Keep signal counts separate from noisy camera intensities. |
| [train.ipynb](../train.ipynb) | Train and validate the segmentation network, then inspect binary pixel metrics. The thesis configuration is documented in docs/research.md; existing outputs are historical. Evaluation loaders here drop incomplete batches; use a complete deterministic loader for any new reported evaluation. |
| [metrics.ipynb](../metrics.ipynb) | Explore the threshold tradeoff between missed signal energy and pixel retention, inspect events and plot distributions. Compare the selected checkpoint and split with the thesis before attributing numerical outputs to Table 4.3. |
| [metrics_minimal.ipynb](../metrics_minimal.ipynb) | Explore compact evaluation and storage representations. Experimental branches use different datasets and normalization settings; actual encoded size is distinct from pixel retention. |
| [midas_test.ipynb](../midas_test.ipynb) | Illustrate application to real MIDAS camera frames. Requires optional DAQ dependencies, a local file and a trusted checkpoint. No ground truth is provided: these examples do not measure real-data accuracy. |
| [main.ipynb](../main.ipynb) | Earlier training experiment. Retained for provenance; start with train.ipynb for the later workflow. |
| [ShowImages.ipynb](../ShowImages.ipynb) | Inspect simulated ROOT images and signal maps. |
| [bench.ipynb](../bench.ipynb) | Standalone network speed experiment. Timings depend on hardware, model and warm-up conditions; they are not automatically comparable to thesis benchmarks. |
| [dataloader.ipynb](../dataloader.ipynb) | Prototype data loading and a small network. Retained as an exploratory notebook. |
| [dataset_stats.ipynb](../dataset_stats.ipynb) | Compute image and target means for a selected split and mask variant. |
| [example_picture.ipynb](../example_picture.ipynb) | Explore experiment-specific CYGNO/MIDAS and PyROOT access. Some cells fetch external data and require collaboration software. |
| [imageio_test.ipynb](../imageio_test.ipynb) | Small PNG I/O experiment. |
| [network_drawing.ipynb](../network_drawing.ipynb) | Generate an architectural visualization using optional visualtorch. Locally defined drawing blocks simplify the network and are not the authoritative model implementation. |
| [testbed.ipynb](../testbed.ipynb) | Scratch experiments and qualitative inspections, including a recorded error. Not an automated test suite; use tests/ for reproducible checks. |

`bak/ShowImages.bak.ipynb` is a historical backup, outside the supported workflow.
