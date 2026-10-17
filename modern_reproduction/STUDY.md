# Diagnostic study (working branch only)

The 2019 TempCNN paper evaluates approximately one million satellite time series. Its GitHub repository includes just 260 example-training and 260 example-test observations. The 2.49-million-parameter model is not expected to reproduce the full-paper accuracy on that sample.

The first real-data example run stopped after two epochs (27/260 and 29/260 correct for the historical and train-only preprocessing protocols). Its 13 validation observations spanned only five classes. Those results are not evidence against TempCNN.

This branch adds a **diagnostic, not a paper reproduction**: three seeded, polygon-disjoint class-balanced holdouts (4 observations per class), up to 50 epochs with patience 10, two otherwise identical scaling protocols, and a Euclidean 1-nearest-neighbour control. The test labels are never used for training or checkpoint selection. Report classification and polygon-majority accuracy, macro-F1, and a paired test-polygon bootstrap interval conditional on each pair of fitted models. Interpret the bootstrap cautiously: it does not quantify variation across datasets or account for choosing this protocol after the first run.

The workflow is triggered only for pushes to this research branch (and can also be started manually). The experiment leaves all original project files untouched. Results are uploaded as a short-lived artifact, including the public upstream CSVs so the findings can be independently audited.

**Not for upstream submission until the findings are reviewed.**
