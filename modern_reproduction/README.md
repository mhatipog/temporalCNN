# TempCNN: repeatable example experiment

This directory independently reconstructs the **example** `Archi_3CONV64_1FC256`
from [Pelletier, Webb & Petitjean (2019)](https://doi.org/10.3390/rs11050523),
using Keras 3 with a PyTorch backend. The historical scripts remain unchanged.
The upstream example data are only a small public subset of the full study.
**Do not compare these accuracies to published full-benchmark results.**

The published example fits its 2nd/98th percentile band scaler before splitting
out validation polygons. This experiment compares that original **pre-split**
protocol with **train-only** scaling, using identical seeded polygon groups.
The change is methodological: it does not establish an error in the paper.

Run from the repository root (Python 3.13; Keras 3; CPU PyTorch):

```sh
pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
pip install 'keras>=3.10,<4' 'numpy>=2,<3'
KERAS_BACKEND=torch python modern_reproduction/example.py --protocol pre-split --output outputs/pre-split
KERAS_BACKEND=torch python modern_reproduction/example.py --protocol train-only --output outputs/train-only
```

Each run checks the Git blob IDs of both original CSVs, fixes the random seed,
records the validation polygon identifiers, and saves class predictions, a
confusion matrix, checkpoint, and JSON report. Exact numerical agreement with
the legacy Keras training run is not expected across versions or backends.

The workflow `Reproduce TempCNN example (manual)` runs both variants on demand,
not on pushes or external pull requests. It uses public repository data and
can be disabled without affecting any historical code.
