# DL-PDAC Paper Artifact

This repository contains code and processed data for the DL-PDAC study:

**Distributed Safe Multi-Agent Reinforcement Learning under Sparse Communication via Lyapunov Primal-Dual Actor-Critic**

The package is a lightweight research artifact with source code, processed tables, and rendered figures.

## Repository Layout

```text
src/                 Core environments, DL-PDAC, and baseline implementations.
experiments/         Training, evaluation, and analysis entry points used for the reported experiments.
data/processed/      Processed table, figure, residual-audit, and diagnostic data.
figures/             Rendered figures used in the manuscript.
scripts/             Lightweight repository checks.
```

## Included Data

Included:

- Navigation results for `n=4`, `n=8`, and `n=20`.
- Baseline comparisons with MADDPG-CBF, HATRPO, PPO PID-Lagrangian, and the aligned Scal-MAPPO-L reproduction.
- Communication-range and fixed-filter controlled comparisons.
- Warehouse stress-test table and warehouse communication-throughput figure data.
- Residual audit, controlled-comparison statistics, averaged-dual drift diagnostics, and controller-side timing values referenced by the manuscript.

Large artifacts such as raw training logs and checkpoints are not part of this lightweight release.

See `data/processed/paper_data_manifest.csv` for a file-by-file map of the processed data files.

## Example Commands

Navigation DL-PDAC run:

```bash
python experiments/train_revision_nav_v2.py --n_agents 20 --n_episodes 1500 --k_hops 1 --dual_mode local --topology chain --seed 0 --device cuda
```

Aligned Scal-MAPPO-L reproduction:

```bash
python experiments/train_scal_mappo_l.py --n_agents 20 --n_episodes 1500 --k_hops 1 --topology chain --cost_mode magnitude --cost_limit 0.05 --seed 0 --device cuda
```

Warehouse sparse DL-PDAC stress test:

```bash
python experiments/train_warehouse_benchmark_v2.py --algorithm dlpac_sparse_k2 --n_agents 10 --n_episodes 500 --seed 0 --device cuda
```

## Verification

Run:

```bash
python scripts/check_release.py
```

The script checks key processed values.
