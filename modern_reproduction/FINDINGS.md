# Re-running the public TempCNN example — observations

17 October 2026. Research notes, **not** results from the full 2019 benchmark.

Pelletier, Webb and Petitjean (2019), *Remote Sensing* 11(5):523.
https://doi.org/10.3390/rs11050523

The accompanying GitHub repository contains 260 training and 260 test
observations (20 per class), each with 149 dates and three spectral bands.
The published study used approximately one million observations. This public
example is therefore useful for executing the model and checking methods,
not for reproducing the paper's reported accuracy.

## Initial behavior

The original five-percent polygon validation split left 13 validation
observations spanning only five of the 13 classes. With zero stopping
patience, the modern equivalent of the original 64-filter model stopped
after two epochs. Test results were 27/260 correct with pre-validation-split
normalization and 29/260 with training-only normalization.

[Original GitHub run](https://github.com/mhatipog/temporalCNN/actions/runs/39025340005).

## Three paired random seeds, exploratory balanced holdout

We held out four observations per class (52 total), keeping polygons intact;
trained at most 50 epochs with ten-epoch stopping patience; and changed
only how the 2nd/98th percentile scaling parameters are estimated.
The reference column uses Euclidean 1-nearest-neighbour on the same training
subset, not the paper's published Random Forest.

| Seed | Scaling before validation split | Training-only scaling | 1-NN, training-only |
| --- | ---: | ---: | ---: |
| 7 | 70.00% | 69.62% | 63.46% |
| 42 | 66.54% | 66.92% | 63.85% |
| 101 | 64.62% | 63.46% | 65.38% |

Average train-only minus pre-split difference across the three seeds:
**−0.38 percentage points**. It is smaller than seed-to-seed variation.
The three paired test-polygon cluster bootstrap intervals do not establish
a stable, meaningful advantage for either scaling procedure. These are
descriptive intervals conditional on fitted models and do not cover all
sources of uncertainty. All six neural runs reached 50 epochs.

[Three-seed study and downloadable predictions](https://github.com/mhatipog/temporalCNN/actions/runs/39026097630).

## Separating stopping patience from validation composition

A separate, single-seed (42) local factorial rerun used training-only
normalization and the **same published CSVs**. CPU PyTorch 2.10.0,
Keras 3.13.2, NumPy 2.3.5, Python 3.13.5.

| Validation set | Patience 0 | Patience 10 |
| --- | ---: | ---: |
| 5% polygon holdout (13 observations) | 11.15% (2 epochs) | 68.46% (39 epochs) |
| Class-balanced 20% holdout (52 observations) | 15.77% (2 epochs) | 66.54% (50 epochs) |

Thus the low original-example score is strongly associated with **premature
stopping** on a very small validation set, rather than a demonstrated defect
in the TempCNN architecture or a large preprocessing-leakage effect. The
different validation partitions also leave different numbers of training
examples; neither split should be presented as the paper's full protocol.

To repeat a factorial cell, run the existing `modern_reproduction/example.py`
with `--validation-split original-5pct` or `balanced-20pct`,
`--patience 0` or `10`, `--epochs 50`, `--protocol train-only`, and
`--seed 42` (using the dependencies described in README.md).

## Interpretation / possible contribution

1. Preserve the historical scripts and accurately describe the small
   example's limitations.
2. Offer a practical, separate example with documented validation and
   stopping settings. Do **not** recommend altering the published protocol
   based only on these exploratory results.
3. Further experiments should pre-specify choices and evaluate on additional
   genuinely independent data; repeated adjustment to this example test set
   cannot establish generalization.
4. Separately, `deeplearning/architecture_complexity.py` repeats selector
   key `3`, making the 128-filter width unreachable through the original
   selector. This should be evaluated independently, including compatibility
   with existing numeric architecture choices.

These findings remain on a research branch. No upstream PR has been sent.
