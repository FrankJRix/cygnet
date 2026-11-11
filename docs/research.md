# Scientific reference

Source: Franco Risi, *Metodi di Machine Learning applicati all’esperimento CYGNO
per la ricerca della materia oscura*, 2025/2026, principally Chapter 4,
Table 4.3 and Appendices A–B. The supplied PDF is a reference document and is not
redistributed in this repository.

## Dataset and objective

The thesis uses 7,200 Geant4-simulated and digitized events in nine groups:
nuclear recoils at 5/15/50 keV, electronic recoils at 1/5/15/50 keV, and two
long-track configurations. The 1 keV electronic recoils and one long-track group
use low gain. Each group contributes 600 training, 100 validation and 100 test
events, for 5,400/900/900 total (75/12.5/12.5%). Simulation and detector context
are described in the thesis; this repository does not implement Geant4 simulation.

Targets identify signal pixels. After max-pool downsampling, square dilation
with radius four (9 × 9 kernel, stride one) creates conservative training regions.
The final mask selects original pixel values; it does not reconstruct intensities.

## Thesis configuration

| Parameter | Value |
| --- | --- |
| Model | `SkipNet(base=16)` |
| Active encoder widths | 16, 32, 64, 128 |
| Input | Single-channel, average pooled by 4 per axis |
| Target | Max pooled by 4, then dilated by radius 4 |
| Loss | Class-balanced sigmoid focal loss, alpha=0.25, gamma=2 |
| Optimizer | AdamW, learning rate 0.01, weight decay 0.01 |
| Scheduler | StepLR, step_size=3, gamma=0.5 |
| Training | 16 epochs, batch size 32 |
| Selected checkpoint | Epoch 15, by validation loss |
| Decision threshold | 0.15, selected on validation EM/PR tradeoff |

`SkipNet` returns logits. Apply sigmoid before thresholding. Decoder blocks
unpool **before** their convolutions, as implemented in `forward`. Skip connections
add pooled encoder features rather than concatenating channels. A fifth level is
registered but inactive; removing it would break historical state dictionaries.
`Net` is a separate three-level experiment, not the final thesis architecture.

The legacy `focal_loss` uses a constant alpha multiplier for both classes;
`binary_focal_loss` and torchvision's `sigmoid_focal_loss` use class-dependent
alpha weights. Preserve this distinction when comparing experiments.
`IoUMetric`/`IoULoss` are legacy three-class utilities, not binary evaluation helpers.

## Physics metrics and limitations

For an event with nonzero true signal, missed energy is
`100 * (true_signal_counts - retained_signal_counts) / true_signal_counts`.
It is an event-wise fraction of simulated signal counts, not the noisy camera
intensity sum. Pixel retention is `100 * selected_pixels / total_pixels`.
The thesis reports means of event-wise fractions, not a dataset-wide energy ratio.

At threshold 0.15, Table 4.3 reports EM=(0.12±0.11)% and PR=(0.38±0.02)%.
Both uncertainties are on the mean. Precision/recall/IoU/Dice and timing
uncertainties instead represent standard deviations (see README for values).
The 17.6 ms network and 47 ms pipeline timings are hardware-specific Tesla T4
measurements; the CPU demo is not a reproduction of these benchmarks.

893/900 frames have EM=0. Seven lose some energy, all in low-gain configurations.
Frame 816 loses all signal; frame 741 loses about 2.98%; frame 762 loses about
1.08%. Appendix A also shows a successfully recovered weak event (frame 824).
These cases should accompany aggregate metrics in any presentation.
Appendix B applies the model to real LIME data without ground truth and supports
qualitative inspection only, not measured real-data accuracy or online deployment.

## Reproduction status

The historical notebooks include configurations and event numbering from multiple
experiment stages. Some use evaluation loaders with `shuffle=True` and
`drop_last=True`; this excludes a partial batch and should not be used for a new
full-test evaluation. A fresh evaluation should use `shuffle=False` and
`drop_last=False`, record event IDs, and report its results separately.

Exact thesis reproduction requires the original split manifest, image data,
checkpoint, preprocessing conventions and hardware. These artifacts are absent
from this checkout. Defaults must not be silently changed to make historical
outputs appear newly reproduced. In particular, preserve the original uint16
image scaling and mask dilation until checked against the checkpoint recipe.
