# HoRM Classify Version Guide (v1-v4)

This document is designed for GitHub and explains the purpose, differences, and usage recommendations of the four classification scripts.

## Table of Contents

- [Files](#files)
- [Shared Pipeline](#shared-pipeline)
- [Version Comparison](#version-comparison)
- [Detailed Notes by Version](#detailed-notes-by-version)
- [Which Version Should You Use](#which-version-should-you-use)
- [Environment Setup](#environment-setup)
- [Common Dataset IDs](#common-dataset-ids)
- [Quick Run](#quick-run)
- [Notes](#notes)

## Files

- [tensor_classify_v1.py](./tensor_classify_v1.py)
- [tensor_classify_v2.py](./tensor_classify_v2.py)
- [tensor_classify_v3.py](./tensor_classify_v3.py)
- [tensor_classify_v4.py](./tensor_classify_v4.py)

## Shared Pipeline

All four versions follow the same core workflow:

1. Load multi-view data and build graph structures using features_to_Lap.
2. Train a semi-supervised classifier with TrustworthyNet_tensor.
3. Evaluate ACC, F1_macro, and F1_micro.
4. Save results to Tensor_classify.txt.

Dataset path setting:

- All four scripts now use the same relative dataset directory: `./datasets/`.

## Version Comparison

| Version | Main Goal | Key Additions | Typical Outputs |
|---|---|---|---|
| v1 | Baseline classification | Standard training and evaluation pipeline | Console metrics + Tensor_classify.txt |
| v2 | Baseline + ablation | Keeps standard training and provides loss ablation toggles | Metrics + Tensor_classify.txt |
| v3 | Baseline + 3D t-SNE visualization | Adds 3D t-SNE feature visualization with improved code structure | Metrics + 3D t-SNE figures |
| v4 | Convergence analysis | Records loss/test acc/train acc by epoch and plots convergence curves | Metrics + Results/Convergence_Curves |

## Detailed Notes by Version

### 1. tensor_classify_v1.py (Baseline)

What it mainly does:

- Provides the standard semi-supervised training and evaluation process.
- Uses supervised loss plus a consistency term (entropy regularization).
- Includes 2D visualization options with UMAP/t-SNE.

Best for:

- Reproducing baseline results.
- Serving as the control version for later comparisons.

### 2. tensor_classify_v2.py (Baseline + Ablation)

What it mainly does:

- Uses standard semi-supervised training as default (supervised + consistency losses).
- Keeps simple ablation toggles for loss terms (you can disable supervised or consistency terms by uncommenting the provided lines).
- Does not include 3D visualization.

Best for:

- Studying the contribution of different loss components.
- Running fair comparisons with a clean training-only pipeline.

### 3. tensor_classify_v3.py (Baseline + 3D t-SNE Visualization)

What it mainly does:

- Extends the baseline training pipeline (same as v1) with a 3D t-SNE visualization of fused multi-view features after each repeat.
- Saves color-coded 3D scatter plots to `./Results/Visualizations_tsne/`.
- Improves code organization and adds clearer version annotations.

Best for:

- Generating presentation-ready visual outputs alongside experiment metrics.
- Inspecting cluster separability of fused features in 3D space.

### 4. tensor_classify_v4.py (Convergence Curves)

What it mainly does:

- Records loss, test accuracy, and training accuracy for each epoch.
- Adds plot_convergence_curves to generate dual-axis convergence plots.

Best for:

- Analyzing training stability, convergence speed, and overfitting trends.
- Creating training-process figures for papers and reports.

## Which Version Should You Use

- Need a stable baseline only: choose v1.
- Need baseline training with easy ablation toggles (no 3D visualization): choose v2.
- Need baseline training with 3D t-SNE feature visualization: choose v3.
- Need training-dynamics and convergence analysis: choose v4.

## Environment Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Make sure the required `.mat` dataset files are available under `./datasets/`.

## Common Dataset IDs

Some commonly used dataset IDs in the scripts are:

| Dataset ID | Dataset Name |
|---|---|
| 1 | ALOI |
| 4 | Caltech101-7 |
| 9 | GRAZ02 |
| 11 | HW |
| 15 | NUS-WIDE |
| 17 | ORL |
| 23 | Youtube |
| 30 | BBC3view |
| 32 | 3sources |
| 35 | MSRC-v1 |
| 36 | NGs |

## Quick Run

Minimal reproducible examples:

```bash
python tensor_classify_v1.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v2.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v3.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v4.py --dataset_id 1 --gamma 0.1 --num_repeats 10
```

You can replace `1` with another valid dataset ID defined in each script's `dataset_name_map`.

Current default settings in the scripts:

- `data_path=./datasets/`
- `seed=42`
- `fusion_type=weight`
- `gamma` is controlled by `--gamma` (default: `0.1`)
- `num_repeats` is controlled by `--num_repeats` (default: `10`)
- `dataset_id` can be passed from the command line with `--dataset_id`
- if `--dataset_id` is omitted, each script falls back to its built-in default dataset list

Typical outputs:

- Quantitative results are appended to `Tensor_classify.txt`
- v3 saves 3D t-SNE figures to `./Results/Visualizations_tsne/`
- v4 saves convergence plots to `./Results/Convergence_Curves/`

## Notes

- Dataset path is unified in all four versions as `./datasets/`.
- Make sure your dataset files are placed under `./datasets/`.
- Typical dependencies include torch, numpy, scikit-learn, matplotlib, umap-learn, and seaborn.
- Use v1 for baseline experiments, then switch to v2/v3/v4 based on your analysis objective.
