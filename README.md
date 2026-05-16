# DL-PDAC Paper Artifact

This repository contains the code and paper-authoritative processed data for the current DL-PDAC manuscript:

**Distributed Safe Multi-Agent Reinforcement Learning under Sparse Communication via Lyapunov Primal-Dual Actor-Critic**

The package is intentionally curated for the reported manuscript. Historical logs, exploratory experiments, checkpoints, and LaTeX build files are not included.

## Repository Layout

```text
src/                 Core environments, DL-PDAC, and baseline implementations.
experiments/         Training, evaluation, and analysis entry points used for the reported experiments.
data/processed/      Canonical paper values for tables, figures, residual audits, and diagnostics.
figures/             Rendered figures used in the manuscript.
scripts/             Lightweight repository checks.
```

## Scope of Included Data

`data/processed/` is the canonical data source for this release. When historical log summaries differed slightly from the manuscript, the manuscript values were treated as authoritative and written into the processed files.

Included:

- Navigation results for `n=4`, `n=8`, and `n=20`.
- Baseline comparisons with MADDPG-CBF, HATRPO, PPO PID-Lagrangian, and the aligned Scal-MAPPO-L reproduction.
- Communication-range and fixed-filter controlled comparisons.
- Warehouse stress-test table and warehouse communication-throughput figure data.
- Residual audit, controlled-comparison statistics, averaged-dual drift diagnostics, and controller-side timing values referenced by the manuscript.

Large artifacts such as raw training logs, checkpoints, and historical intermediate summaries are not part of this lightweight release.

See `data/processed/paper_data_manifest.csv` for a file-by-file map from manuscript item to processed data file.

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

The code was developed with PyTorch, Gymnasium, PettingZoo, NumPy, SciPy, and Matplotlib. GPU execution is recommended for full training runs.

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

TeamComm is an external baseline. To run that baseline, clone TeamComm separately and set:

```bash
export TEAMCOMM_REPO_ROOT=/path/to/TeamComm
```

On Windows PowerShell:

```powershell
$env:TEAMCOMM_REPO_ROOT = "C:\path\to\TeamComm"
```

## Verification

Run:

```bash
python scripts/check_release.py
```

The script checks key processed values against the manuscript.
