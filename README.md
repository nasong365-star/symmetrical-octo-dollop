
# HoRM_v6.0

HoRM_v6.0 is a research codebase for multi-view learning experiments, including semi-supervised classification and clustering.

## Contents

- Classification scripts: `tensor_classify_v1.py` to `tensor_classify_v4.py`
- Clustering scripts: `tensor_clustering_v1.py` (baseline), `tensor_clustering_v2.py` (ablation)
- Core model: `TrustworthyNet.py`
- Dataset utilities: `util/loadMatData.py`

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Place the required `.mat` dataset files under `./datasets/`.

## Quick Start

Demo setup in this repository uses one dataset per task:

- Classification demo dataset: `ALOI` (`dataset_id=1`)
- Clustering demo dataset: `3sources` (`dataset_clustering_id=3`)

Run a classification experiment:

```bash
python tensor_classify_v1.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v2.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v3.py --dataset_id 1 --gamma 0.1 --num_repeats 10
python tensor_classify_v4.py --dataset_id 1 --gamma 0.1 --num_repeats 10
```

Run a clustering experiment:

```bash
python tensor_clustering_v1.py --dataset_clustering_id 3 --clustering_seed 50
python tensor_clustering_v2.py --dataset_clustering_id 3 --clustering_seed 50
```

## Documentation

- Classification guide: [README_classify.md](./README_classify.md)
- Python dependencies: [requirements.txt](./requirements.txt)
- License: [LICENSE](./LICENSE)

## Notes

- Classification scripts support `--dataset_id` for command-line dataset selection.
- Classification scripts also support `--gamma` and `--num_repeats` from the command line.
- If `--dataset_id` is omitted, each classify script falls back to its built-in default dataset list.
- Clustering scripts support `--dataset_clustering_id` and `--clustering_seed` from the command line.
- Clustering v1 is baseline (`Self1 + Self2 + consistency`), and v2 is the ablation variant.
- Classification metrics are appended to `Tensor_classify.txt`.
- Clustering metrics are appended to `Tensor_cluster.txt`.

