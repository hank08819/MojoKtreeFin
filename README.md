# MojoKTree: SIMD K-D Tree KNN for Financial Time Series

**Paper:** *Fast Exact Nearest-Neighbor Learning for High-Frequency Financial Time Series*  
**Authors:** Henry Han, Diane Li
**Code and data:** https://github.com/hank08819/MojoKtreeFin

---

## What This Is

A Mojo SIMD k-d tree for exact KNN inference on large financial feature corpora.  
Core design: variance-based splitting + contiguous flat-buffer layout + compile-time float32 SIMD distance kernels.

**Key results (vs. scikit-learn k-d tree):**
- x86: 17.5–21.6× speedup across CPRI, JPM, WMT, AAPL (up to 277K training samples)
- ARM64 (Apple M3): 28.1–43.5× over sklearn brute force on BAC, SPY, QQQ

---

## Repository Structure

```
MojoKTree/
├── mojo/
│   ├── kd_tree_knn.mojo       — Main SIMD k-d tree KNN benchmark (d=24, ARM64)
│   ├── extra_trees_iv.mojo    — Extra Trees IV regression, mixed calls+puts (d=8)
│   └── extra_trees_iv7.mojo   — Extra Trees IV regression, calls-only / puts-only (d=7)
├── python/
│   └── sklearn_baseline.py    — scikit-learn KNN baseline for comparison
├── demo_data/
│   └── EURUSD/                — Demonstration dataset: EURUSD FX, d=24, ~64K samples
│       ├── train_X.bin        — Feature matrix (float32, row-major)
│       ├── train_y.bin        — Direction labels (int32)
│       ├── test_X.bin
│       └── test_y.bin
└── README.md
```

---

## Quick Start

### Requirements
- Mojo 25.1.1 via `conda` (Magic / Modular toolchain)
- Python 3.12 + scikit-learn 1.5.0 (for baseline comparison)

### Run the KD-Tree KNN demo on EURUSD

```bash
# Update DATA_DIR at the top of the file to point to demo_data/
conda run -n base mojo mojo/kd_tree_knn.mojo
```

Expected output on ARM64 (Apple M3):
```
=== EURUSD ===
  n_train: 63963   n_test: 13692
  Build: ~38 ms
  Mojo KD mean: ~0.209 s   acc: ~0.503
```

### Run the scikit-learn baseline

```bash
python python/sklearn_baseline.py
```

### Run Extra Trees IV regression

```bash
# Mixed calls+puts (d=8), 200K train
conda run -n base mojo mojo/extra_trees_iv.mojo

# Calls-only and puts-only (d=7), 200K train each
conda run -n base mojo mojo/extra_trees_iv7.mojo
```

---

## Binary Data Format

All Mojo files read data in this compact binary format:

**Feature matrix X** (`train_X.bin`, `test_X.bin`):
```
[n_rows : int32][n_cols : int32][float32 × n_rows × n_cols, row-major]
```

**Labels y — classification** (`train_y.bin`, `test_y.bin`):
```
[n_rows : int32][int32 × n_rows]
```

Values: 1 = next-minute price up, 0 = price down or flat.

---

## Generating Your Own Dataset

To prepare binary data from OHLCV CSV:

```bash
# 1. Fetch minute-bar data (yfinance / Tiingo)
python python/fetch_equity.py        # in full code/ directory

# 2. Enrich with 24 technical features
python python/enrich_d24.py

# 3. Normalize and convert to binary
python python/normalize.py
python python/prepare_binary.py
```

See the full `code/python/` directory for complete pipeline scripts.

---

## Implementation Notes

- SIMD width: `comptime SW = 8` (optimal for x86 256-bit; use SW=4 for ARM64 128-bit native)
- Feature dimension: `comptime DIMS = 24` (change to 16 for x86 datasets)
- Leaf size: `comptime LEAF = 10`
- KNN neighbors: `comptime KN = 5`
- Split axis: highest-variance dimension at each node
- Storage: flat row-major float32 buffer with permuted index array

---

## Citation

```bibtex
@inproceedings{han2026mojo,
  title={Fast Exact Nearest-Neighbor Learning for High-Frequency Financial Time Series},
  author={Han, Henry and Li, Diane},
  booktitle={Lecture Notes in Computer Science},
  year={2026},
  publisher={Springer}
}
```
