#!/usr/bin/env python3
"""Gurobi 14.0.0beta1: incumbent accepted although it violates lazy constraints.

Self-contained reproduction (needs only gurobipy). The license is taken from
the normal lookup, e.g.  GRB_LICENSE_FILE=/path/to/gurobi.lic python lazy_cut_repro.py

Model (9x9 Sudoku clue minimization, 1458 binaries):
  y[r,c,k] = 1 if cell (r,c) of the solution grid holds digit k (Sudoku rules)
  x[r,c,k] = 1 if that entry is shown as a clue, x <= y, minimize sum(x)

In a MIPSOL callback every candidate (y*, x*) is checked by a separate small
Gurobi model that searches for a second Sudoku z != y* consistent with the
clues x*. For each such z, with D = {cells where y* and z differ}, the lazy cut

    sum_{(r,c) in D} ( x[r,c,y*_rc] - y[r,c,y*_rc] ) >= 1 - |D|

is added with cbLazy. At the candidate its left-hand side is -|D|, so every
added cut is violated by exactly 1 and the candidate must be rejected.

Expected: Gurobi never accepts a solution that violates an added lazy cut.
Observed with 14.0.0beta1: status OPTIMAL with objective 0. The accepted
incumbent was passed to MIPSOL and cut off, yet it is returned as optimal;
it violates the stored lazy cuts by 1.0. Gurobi 13.0.2 on the same script
does not accept it (it runs until the time limit).
"""

import math
import platform
import sys

import gurobipy as gp
from gurobipy import GRB

N = 9
B = math.isqrt(N)
R = range(N)
TIME_LIMIT = 120


def sudoku(model, v):
    model.addConstrs(v.sum(r, c, "*") == 1 for r in R for c in R)
    model.addConstrs(v.sum(r, "*", k) == 1 for r in R for k in R)
    model.addConstrs(v.sum("*", c, k) == 1 for c in R for k in R)
    for br in range(B):
        for bc in range(B):
            cells = [(r, c) for r in range(br * B, br * B + B) for c in range(bc * B, bc * B + B)]
            model.addConstrs(gp.quicksum(v[r, c, k] for r, c in cells) == 1 for k in R)


def grid(vals):
    return {(r, c): max(R, key=lambda k: vals[r, c, k]) for r in R for c in R}


def main():
    print("gurobipy", gp.gurobi.version(), "|", platform.platform(), "| python", platform.python_version())

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()

        # Master.
        m = gp.Model("master", env=env)
        y = m.addVars(N, N, N, vtype=GRB.BINARY, name="y")
        x = m.addVars(N, N, N, vtype=GRB.BINARY, name="x")
        sudoku(m, y)
        m.addConstrs(x[r, c, k] <= y[r, c, k] for r in R for c in R for k in R)
        for c in R:
            y[0, c, c].LB = 1  # symmetry breaking: first row is 1..9
        m.setObjective(x.sum(), GRB.MINIMIZE)
        m.Params.OutputFlag = 1
        m.Params.LazyConstraints = 1
        m.Params.MIPGap = 0
        m.Params.TimeLimit = TIME_LIMIT

        # Separation model: another Sudoku z consistent with the clues, z != y*.
        s = gp.Model("separation", env=env)
        z = s.addVars(N, N, N, vtype=GRB.BINARY, name="z")
        sudoku(s, z)
        s.Params.DualReductions = 0

        lazy_cuts = []       # (list of (r, c, k) in D, rhs) for later checking
        rejected = set()     # candidates that were cut off in MIPSOL
        stats = {"mipsol": 0, "cuts": 0}

        def callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return
            stats["mipsol"] += 1
            ys, xs = grid(model.cbGetSolution(y)), model.cbGetSolution(x)
            for r in R:
                for c in R:
                    for k in R:
                        z[r, c, k].LB = 1 if xs[r, c, k] > 0.5 else 0
            temp = [s.addConstr(gp.quicksum(z[r, c, ys[r, c]] for r in R for c in R) <= N * N - 1)]
            for _ in range(4):
                s.optimize()
                if s.Status != GRB.OPTIMAL:
                    break
                zs = grid(s.getAttr("X", z))
                D = [(r, c, ys[r, c]) for r in R for c in R if ys[r, c] != zs[r, c]]
                model.cbLazy(gp.quicksum(x[i] - y[i] for i in D) >= 1 - len(D))
                lazy_cuts.append((D, 1 - len(D)))
                stats["cuts"] += 1
                rejected.add(tuple(sorted(ys.items())) + tuple(i for i in sorted(xs) if xs[i] > 0.5))
                temp.append(s.addConstr(gp.quicksum(z[r, c, zs[r, c]] for r in R for c in R) <= N * N - 1))
            s.remove(temp)

        m.optimize(callback)

        print(f"\nstatus={m.Status} (OPTIMAL={GRB.OPTIMAL})  SolCount={m.SolCount}  "
              f"ObjVal={m.ObjVal if m.SolCount else None}  ObjBound={m.ObjBound}")
        print(f"MIPSOL callbacks={stats['mipsol']}  lazy cuts added={stats['cuts']}")
        if not m.SolCount:
            print("No incumbent; not reproduced.")
            return 0

        yv, xv = m.getAttr("X", y), m.getAttr("X", x)
        ys = grid(yv)
        key = tuple(sorted(ys.items())) + tuple(i for i in sorted(xv) if xv[i] > 0.5)
        worst = max(rhs - sum(xv[i] - yv[i] for i in D) for D, rhs in lazy_cuts)
        violated = sum(1 for D, rhs in lazy_cuts if rhs - sum(xv[i] - yv[i] for i in D) > 1e-6)
        print(f"final incumbent was cut off in MIPSOL before: {key in rejected}")
        print(f"final incumbent violates {violated} of {len(lazy_cuts)} lazy cuts, max violation {worst:.3g}")

        bug = worst > 1e-6
        print("\nREPRODUCED: accepted incumbent violates lazy constraints." if bug
              else "\nNot reproduced: incumbent satisfies all lazy constraints.")
        return 1 if bug else 0


if __name__ == "__main__":
    sys.exit(main())
