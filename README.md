# Sudoku with logic-based Benders decomposition

Generate both a completed Sudoku and the smallest possible subset of its entries
that determines it uniquely. A Gurobi master chooses the grid and clues; a
second integer program finds alternative completions and generates globally
valid cuts. See [the model note](tex/sudoku_benders_model.pdf) and its
[TeX source](tex/sudoku_benders_model.tex).

## Linux: install and run for 12 hours

Use Python 3.11 or newer and a Gurobi license that permits the full model.
The free size-limited license is insufficient for the strengthened 9×9 model.
The package is pinned to **Gurobi 13.0.2**: the repository contains a reproduction
of incorrect lazy-constraint acceptance in 14.0.0beta1. We have not verified a fix
in later version-14 builds. A license for version 13 or newer is needed.

```bash
git clone https://github.com/tprunarsson/sudoku-benders.git
cd sudoku-benders
# If the license is not installed in a standard location:
export GRB_LICENSE_FILE=/path/to/gurobi.lic
bash scripts/setup_linux.sh
```

The setup creates `.venv`, installs dependencies, and checks the license. For
WLS, pass `--wls-file /path/to/credentials.txt` to the run script instead; the
setup's license check may fail until WLS credentials are supplied. Credential
files should be kept outside the repository.

Run inside `tmux` so disconnecting SSH does not stop the experiment:

```bash
tmux new -s sudoku
bash scripts/run_linux.sh
# Detach with Ctrl-b, then d. Reconnect with: tmux attach -t sudoku
```

The default budget is **43,200 seconds total**, including warm-start generation
and verification. Solver shutdown and writing the final result can slightly
exceed the budget. The launcher uses up to 16 available CPUs for the master and
one for the separator. Override these for your machine:

```bash
THREADS=8 SEED=1 bash scripts/run_linux.sh
# Optional memory limit, in GB:
THREADS=8 MEMORY_GB=24 bash scripts/run_linux.sh
# A one-minute installation check:
SECONDS_LIMIT=60 THREADS=4 bash scripts/run_linux.sh
```

Additional Python options can be passed at the end of the launcher command.
For example, `--separator feasible` overrides the default hybrid separator.
`PYTHON=/path/to/python` selects an existing environment; `RUN_DIR=/path/to/new-dir`
selects an empty output directory. The launcher refuses to overwrite an earlier run.

## What the overnight profile enables

- One persistent branch-and-cut search with lazy **unavoidable-set cuts**;
  these have fewer coefficients than the conditional delta formulation.
- First-row digit normalization, row/band ordering, digit coverage, and the
  empty-row/column pair inequalities.
- Four-cell rectangle inequalities in the lazy pool with `Lazy=3`, allowing
  violated root-relaxation inequalities to enter the LP.
- Eight witnesses per candidate. The hybrid separator spends at most 0.5 seconds
  seeking a close alternative, uses any incumbent found, and falls back to
  feasibility if it found neither a witness nor a uniqueness proof.
- A newly generated grid with greedy, uniqueness-preserving clue removal as a
  MIP start. Alternatives encountered during removal also seed Benders cuts.
- `MIPFocus=1` to favor finding feasible puzzles, and `NodefileStart=0.5` to
  spill a large node tree to the run directory. Other search settings stay automatic.
- The published **17-clue lower bound**. This relies on
  [McGuire, Tugemann and Civario's exhaustive proof](https://arxiv.org/abs/1201.0749);
  it saves time trying to rule out smaller puzzles. Use `--min-clues 0` to derive
  a bound using only this model. Fixed grids may require more than 17 clues.

The objective is simply the number of clues. Positional weights and clue-spread
requirements are disabled. The optional iterative bound/start/pruning tricks
remain disabled because the model note's earlier benchmarks found them slower.
The run searches over all target grids rather than fixing a target or top-band class.

These are practical defaults, not a claim of optimal tuning for every Linux
machine. In local 30-second tests with four master threads, the hybrid batch-8
profile found **21 clues**, hybrid batch-1 found 24, and feasibility batch-8
retained 26. Every reported puzzle was independently checked. The latter two
tests ran concurrently, so this is a short screening comparison rather than a
controlled long-run benchmark. The [21-clue example](examples/smoke_9x9.json)
is included for inspection; it is not loaded as the default starting puzzle.

Gurobi documents the settings in its
[parameter guidelines](https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html)
and [lazy-constraint attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html).

## Progress and results

Each experiment creates a directory under `runs/` containing:

| File | Meaning |
| --- | --- |
| `best.json` | Best independently verified unique puzzle and its completed grid; replaced atomically on improvement |
| `progress.json` | Latest elapsed time, lower bound, and best clue count; updated approximately every 30 seconds when solver callbacks run |
| `events.jsonl` | Complete candidate/separation trace, one JSON record per line |
| `environment.json` | Solver/Python versions and experiment settings, excluding credentials |
| `run.log` | Console output and Gurobi log |
| `result.json` | Final status, best verified puzzle if available, bound, counters, and a bounded recent history |
| `nodes/` | Gurobi's disk-backed node storage, if needed |

```bash
cat runs/<run-directory>/progress.json
tail -f runs/<run-directory>/run.log
cat runs/<run-directory>/best.json
```

At a time or memory limit, the final result retains the best verified puzzle
even if the last master candidate is non-unique. A bound of 17 and a puzzle
with 21 clues means the unresolved minimum lies between 17 and 21. A verified
17-clue result attains the known global minimum. The code does not report an
unchecked timeout candidate as a verified puzzle, and callback exceptions stop
the solve instead of silently disabling separation.

`best.json` survives interruption after it has been written. Restart with its
puzzle as the MIP start, retaining the same model/symmetry settings:

```bash
SEED=1 bash scripts/run_linux.sh --start-json runs/<old-run>/best.json
```

This starts a fresh search with the old incumbent; the branch-and-bound tree and
accumulated cuts are not resumed. The checkpoint is checked again before use.

## Small checks and other modes

```bash
.venv/bin/python python/sudoku_benders_gurobi.py --size 4 --warm-start --strengthen --row-symmetry
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Tests compare the independent checker with all 288 completed 4×4 grids and
exercise optimality in both Benders modes, cut families, checkpoint reuse,
callback errors, and timeout behavior. A working Gurobi license is needed for
the MIP integration tests.

The original SciPy/HiGHS implementation is also available:

```bash
.venv/bin/python -m pip install -r requirements-highs.txt
.venv/bin/python python/sudoku_benders_generator.py --size 4
```

Use `python/sudoku_benders_gurobi.py --help` for fixed-grid benchmarks, top-band
classes, spread requirements, iterative mode, and other options. The Linux
launcher treats expected time/memory limits as successful experiment completion;
the underlying Python CLI returns 0 for an optimum, 2 for other solver outcomes,
and 3 for configuration/license errors.
