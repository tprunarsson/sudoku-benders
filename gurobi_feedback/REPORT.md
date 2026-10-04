# Gurobi 14.0.0beta1: incumbent that violates lazy constraints is returned as optimal

## Summary

With `LazyConstraints=1`, Gurobi 14.0.0beta1 returns status `OPTIMAL` with an
incumbent that was passed to the `MIPSOL` callback, rejected there with
`cbLazy`, and violates those lazy constraints by 1.0. Gurobi 13.0.2 does not do
this on the same script.

## Environment

- Gurobi 14.0.0beta build v14.0.0beta1, gurobipy 14.0.0b1, Python 3.14.7
- macOS 27.0 (Darwin 27.0.0, 26A428), Apple M4 Max, 14 threads
- Default parameters apart from `LazyConstraints=1`, `MIPGap=0`, `TimeLimit=120`

## How to run

```
GRB_LICENSE_FILE=/path/to/gurobi.lic python lazy_cut_repro.py
```

The script only needs gurobipy and takes a few seconds. Exit code 1 means the
bug was reproduced.

## Model

The model is a 9x9 Sudoku clue minimization with 1458 binaries: grid variables
`y` under the Sudoku rules, clue variables `x <= y`, and the objective
`min sum(x)`. In `MIPSOL`, a second small model looks for another Sudoku that
agrees with the candidate's clues. For every such alternative the callback
adds a lazy cut that the candidate violates by exactly 1. The script keeps
every cut it adds and, at the end, checks the returned incumbent against all
of them.

## Results

| Version | Threads | Status | Objective | Accepted solution violates lazy cuts? |
|---|---|---|---|---|
| 14.0.0beta1 | 14 (default) | OPTIMAL after 4 s | 0 | **Yes: 8 of 1720 cuts violated, max violation 1.0; this exact solution was cut off in MIPSOL before** |
| 14.0.0beta1 | 14 (repeat) | identical | 0 | identical |
| 14.0.0beta1 | 1 | TIME_LIMIT (120 s), no incumbent | – | No |
| 13.0.2 | 14 (default) | TIME_LIMIT (120 s), obj 42 | 42 | No (0 of 17151) |

The Gurobi 14 log shows the bad solution coming from the tree search, not
from a heuristic:

```
* 1543   998              46       0.0000000    0.00000  0.00%  53.3    4s
```

It reproduces the same way on every multi-threaded run, but not with
`Threads=1`, so it seems to depend on parallel tree search. Possibly a lazy
constraint added from one thread is not applied before another thread's node
solution with the same values is accepted.

## Expected behavior

A solution rejected in `MIPSOL` with `cbLazy` (all of whose added cuts it
violates) must never become the incumbent, and in particular must not be
reported as optimal.

## Attached files

- `lazy_cut_repro.py`: the reproduction script
- `run14_1.log`: full Gurobi 14.0.0beta1 output
- `run13.log`: full Gurobi 13.0.2 output of the same script
