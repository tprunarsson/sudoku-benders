#!/usr/bin/env python3
"""Generate a Sudoku and a minimum clue set with logic-based Benders decomposition.

Gurobi (gurobipy) version of sudoku_benders_generator.py.

The master chooses both:
  y[r,c,k] = 1 if digit k is the generated target value in cell (r,c)
  x[r,c,k] = 1 if that target value is exposed as a clue

The subproblem searches for a second complete Sudoku s satisfying the selected
clues. Two solution modes are available:

  callback  (default) One branch-and-cut run. Every incumbent the master finds
            is checked by the uniqueness subproblem inside a MIPSOL callback,
            and violated Benders cuts are added as lazy constraints.
  iterative The classical restricted-master loop of the original script, but
            with a persistent Gurobi master that is re-optimized after cuts
            are added instead of being rebuilt from scratch.

Two cut families are available:

  delta       The conditional cut from the model note: if the target is not s,
              at least one clue must contradict s. In iterative mode this uses
              the delta_s variable exactly as in the note. In callback mode
              variables cannot be added, so delta_s is projected out:
                sum_{r,c,k != s_rc} x_rck + y_{i,j,s_ij} >= 1
              added for every cell (i,j) where the incumbent differs from s.
  unavoidable Let D be the cells where the incumbent target t differs from s.
              Any Sudoku that agrees with t on D can have D swapped to the
              values of s and remain valid, so it needs a clue inside D:
                sum_{(r,c) in D} (x_{r,c,t_rc} - y_{r,c,t_rc}) >= 1 - |D|
              This is one short row per alternative and is usually stronger.

Both families are valid for the full problem, so any combination is exact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sudoku_search import find_alternative
from sudoku_run_support import RunRecorder, atomic_json, canonical_rows, greedy_start, meets_spread

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - depends on local installation
    sys.exit("gurobipy is not importable. Install Gurobi's Python package first.")


@dataclass
class BendersResult:
    status: str
    size: int
    mode: str
    cut: str
    target: np.ndarray
    puzzle: np.ndarray
    clue_count: Optional[int]
    lower_bound: Optional[float]
    master_solves: int
    subproblem_solves: int
    cuts: int
    alternatives: int
    subproblem_seconds: float
    elapsed_seconds: float
    verified_unique: Optional[bool]
    solver_version: str
    history: list[dict] = field(default_factory=list)
    best_source: Optional[str] = None
    assumed_min_clues: int = 0

    def as_jsonable(self) -> dict:
        d = dict(self.__dict__)
        d["target"] = self.target.tolist()
        d["puzzle"] = self.puzzle.tolist()
        return d


def box_cells(n: int) -> list[list[tuple[int, int]]]:
    b = math.isqrt(n)
    if b * b != n:
        raise ValueError("N must have an integer square root, e.g. N=4 or N=9.")
    return [
        [
            (r, c)
            for r in range(br * b, (br + 1) * b)
            for c in range(bc * b, (bc + 1) * b)
        ]
        for br in range(b)
        for bc in range(b)
    ]


def add_sudoku_constraints(model: gp.Model, v: gp.tupledict, n: int, name: str) -> None:
    """Add the four standard Sudoku constraint families on one-hot variables v."""
    N = range(n)
    model.addConstrs((v.sum(r, c, "*") == 1 for r in N for c in N), name=f"{name}_cell")
    model.addConstrs((v.sum(r, "*", k) == 1 for r in N for k in N), name=f"{name}_row")
    model.addConstrs((v.sum("*", c, k) == 1 for c in N for k in N), name=f"{name}_col")
    for b, cells in enumerate(box_cells(n)):
        model.addConstrs(
            (gp.quicksum(v[r, c, k] for r, c in cells) == 1 for k in N),
            name=f"{name}_box{b}",
        )


def to_array(values: dict, n: int) -> np.ndarray:
    out = np.zeros((n, n, n))
    for (r, c, k), val in values.items():
        out[r, c, k] = val
    return out


def decode(yv: np.ndarray, xv: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    target = np.argmax(yv, axis=2)
    clues = (xv > 0.5).astype(np.int8)
    return target, clues, int(clues.sum())


def clue_grid(n: int, clues: np.ndarray) -> np.ndarray:
    """Return an N x N puzzle with 0 for blanks and 1..N for clues."""
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            ks = np.where(clues[r, c, :] > 0)[0]
            if len(ks):
                out[r, c] = int(ks[0]) + 1
    return out


# Built-in grids for --fixed-grid. Blanks (0) are completed by a Sudoku solve.
FIXED_GRIDS = {
    # The 17-clue puzzle of Fig. 1 in McGuire, Tugemann & Civario (arXiv:1201.0749).
    # Its solution grid therefore has minimum clue count exactly 17.
    "mcguire17": (
        "000801000" "000000043" "500000000"
        "000070800" "000000100" "020030000"
        "600000075" "003400000" "000200600"
    ),
}


def parse_grid(spec: str, n: int) -> np.ndarray:
    """Parse a built-in name, a file, or a literal string of N*N digits (0 or . = blank)."""
    if spec in FIXED_GRIDS:
        text = FIXED_GRIDS[spec]
    elif Path(spec).expanduser().is_file():
        text = Path(spec).expanduser().read_text()
    else:
        text = spec
    chars = [ch for ch in text if ch.isdigit() or ch == "."]
    if len(chars) != n * n:
        raise ValueError(f"grid {spec!r} has {len(chars)} cells, expected {n * n}")
    grid = np.array([0 if ch == "." else int(ch) for ch in chars]).reshape(n, n)
    if grid.max() > n:
        raise ValueError(f"grid {spec!r} contains digits larger than {n}")
    return grid


def complete_grid(n: int, env: gp.Env, givens: np.ndarray,
                  time_limit: Optional[float] = None) -> np.ndarray:
    """Return a complete Sudoku (0-based digits) extending givens (1..N, 0 = blank)."""
    with gp.Model("complete", env=env) as model:
        model.Params.Threads = 1
        if time_limit is not None:
            model.Params.TimeLimit = time_limit
        v = model.addVars(n, n, n, vtype=GRB.BINARY)
        add_sudoku_constraints(model, v, n, "v")
        for r in range(n):
            for c in range(n):
                if givens[r, c]:
                    v[r, c, int(givens[r, c]) - 1].LB = 1.0
        model.optimize()
        if model.SolCount == 0:
            raise ValueError("the fixed grid cannot be completed to a valid Sudoku")
        return np.argmax(to_array(model.getAttr("X", v), n), axis=2)


SPEEDUPS = ("bound", "start", "prune")

# The clue count is bounded below by 0, so INF_OR_UNBD can only mean infeasible.
MASTER_INFEASIBLE = (GRB.INFEASIBLE, GRB.INF_OR_UNBD)


def parse_range(text: str) -> tuple[Optional[int], Optional[int]]:
    """'MIN:MAX', 'MIN:', ':MAX' or 'N' (exactly N)."""
    try:
        if ":" not in text:
            return int(text), int(text)
        lo, hi = text.split(":", 1)
        return (int(lo) if lo else None), (int(hi) if hi else None)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected MIN:MAX, MIN:, :MAX or N, got {text!r}") from None


def parse_speedups(text: str) -> frozenset:
    if text in ("none", ""):
        return frozenset()
    if text == "all":
        return frozenset(SPEEDUPS)
    names = frozenset(part.strip() for part in text.split(","))
    unknown = names - set(SPEEDUPS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown speedup(s): {', '.join(sorted(unknown))}")
    return names


def parse_param(text: str) -> tuple[str, object]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    name, raw = text.split("=", 1)
    for conv in (int, float):
        try:
            return name.strip(), conv(raw)
        except ValueError:
            pass
    return name.strip(), raw.strip()


WLS_KEYS = ("WLSACCESSID", "WLSSECRET", "LICENSEID")


def read_wls_file(path: Path) -> dict[str, object]:
    """Read WLSACCESSID, WLSSECRET and LICENSEID from a KEY=VALUE file (e.g. a WLS gurobi.lic)."""
    params: dict[str, object] = {}
    for line in path.expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in WLS_KEYS:
            params[key] = int(value) if key == "LICENSEID" else value
    missing = [k for k in WLS_KEYS if k not in params]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")
    return params


class Subproblem:
    """Persistent uniqueness subproblem: find a Sudoku z != target matching the clues."""

    def __init__(self, n: int, env: gp.Env, time_limit: Optional[float], threads: Optional[int]):
        self.n = n
        self.model = gp.Model("uniqueness", env=env)
        self.model.Params.DualReductions = 0  # report INFEASIBLE, never INF_OR_UNBD
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit
        if threads is not None:
            self.model.Params.Threads = threads
        self.z = self.model.addVars(n, n, n, vtype=GRB.BINARY, name="z")
        add_sudoku_constraints(self.model, self.z, n, "z")
        self.solves = 0
        self.seconds = 0.0

    def _agreement(self, grid: np.ndarray) -> gp.LinExpr:
        n = self.n
        return gp.quicksum(self.z[r, c, int(grid[r, c])] for r in range(n) for c in range(n))

    def solve(
        self,
        target: np.ndarray,
        clues: np.ndarray,
        excluded: Optional[list[np.ndarray]] = None,
        separator: str = "feasible",
        time_limit: Optional[float] = None,
    ) -> tuple[str, Optional[np.ndarray]]:
        """Return ('alternative', s), ('infeasible', None) or ('limit', None)."""
        n, m, z = self.n, self.model, self.z
        excluded = [] if excluded is None else excluded

        if separator == "hybrid":
            # Small difference sets make compact, reusable cuts. A time-limited
            # incumbent is sufficient; a proof of closest distance is unnecessary.
            started = time.perf_counter()
            closest_limit = 0.5 if time_limit is None else min(0.5, time_limit)
            status, alt = self.solve(target, clues, excluded, "closest", closest_limit)
            if status != "limit":
                return status, alt
            left = None if time_limit is None else max(0.0, time_limit - (time.perf_counter() - started))
            if left is not None and left <= 0:
                return "limit", None
            return self.solve(target, clues, excluded, "feasible", left)

        # Enforce selected clues through lower bounds (cheaper than rows).
        for r in range(n):
            for c in range(n):
                for k in range(n):
                    z[r, c, k].LB = 1.0 if clues[r, c, k] > 0 else 0.0

        # Require a solution different from the target and from batch alternatives.
        temps = [m.addConstr(self._agreement(target) <= n * n - 1)]
        temps += [m.addConstr(self._agreement(s) <= n * n - 1) for s in excluded]

        if separator == "closest":
            # Maximize agreement with target, equivalently minimize Hamming distance.
            m.setObjective(self._agreement(target), GRB.MAXIMIZE)
        elif separator == "feasible":
            m.setObjective(0.0)
        else:
            raise ValueError("separator must be 'feasible', 'closest' or 'hybrid'")

        old_limit = m.Params.TimeLimit
        if time_limit is not None:
            m.Params.TimeLimit = time_limit

        t0 = time.perf_counter()
        m.optimize()
        self.seconds += time.perf_counter() - t0
        self.solves += 1

        status, alt = "limit", None
        if m.Status == GRB.INFEASIBLE:
            status = "infeasible"
        elif m.SolCount > 0:
            # A feasible incumbent is enough for a valid cut, even if a
            # closest-solution optimization hit its time limit.
            status = "alternative"
            alt = np.argmax(to_array(m.getAttr("X", z), n), axis=2)

        m.Params.TimeLimit = old_limit
        m.remove(temps)
        return status, alt


class Master:
    """Master model with y (target) and x (clue) variables plus Benders cut helpers."""

    def __init__(
        self,
        n: int,
        env: gp.Env,
        fix_first_row: bool,
        row_symmetry: bool = False,
        digit_cover: bool = False,
        line_pairs: bool = False,
        rectangles: str = "off",
        fixed_grid: Optional[np.ndarray] = None,
        top_band: Optional[np.ndarray] = None,
        prune_dominated: bool = True,
        min_clues: int = 0,
        position_weights: bool = False,
        spread: Optional[dict[str, tuple[Optional[int], Optional[int]]]] = None,
    ):
        self.n = n
        self.min_clues = min_clues
        self.prune_dominated = prune_dominated
        # Unavoidable-set cuts by key {(r, c, target digit)}, for dominance checks.
        self.unavoidable: dict[frozenset, gp.Constr] = {}
        self.model = m = gp.Model("master", env=env)
        self.y = m.addVars(n, n, n, vtype=GRB.BINARY, name="y")
        self.x = m.addVars(n, n, n, vtype=GRB.BINARY, name="x")
        add_sudoku_constraints(m, self.y, n, "y")

        # x_ijk <= y_ijk: a clue can reveal only the generated target value.
        m.addConstrs(
            (self.x[r, c, k] <= self.y[r, c, k] for r in range(n) for c in range(n) for k in range(n)),
            name="clue_link",
        )

        # Fixed-grid mode: the target is given, only the clues are chosen.
        # Symmetry breaking would contradict an arbitrary given grid.
        if fixed_grid is not None:
            for r in range(n):
                for c in range(n):
                    for k in range(n):
                        self.y[r, c, k].LB = self.y[r, c, k].UB = float(fixed_grid[r, c] == k)
            fix_first_row = row_symmetry = False

        # Top-band mode: the first band is fixed to a class representative.
        # This replaces the first-row relabelling; rows inside the fixed band
        # are not reordered below.
        b = math.isqrt(n)
        if top_band is not None:
            for r in range(b):
                for c in range(n):
                    for k in range(n):
                        self.y[r, c, k].LB = self.y[r, c, k].UB = float(top_band[r, c] == k)
            fix_first_row = False

        self._add_strengthening(digit_cover, line_pairs, rectangles)
        self._add_spread(spread or {})

        # Lower bound on the clue count: an assumed minimum (e.g. 17 for 9x9,
        # McGuire et al.) and, with the 'bound' speedup, the last master optimum.
        self.lb_constr = m.addConstr(self.x.sum() >= min_clues, name="objective_bound")

        # Safe symmetry breaking under digit relabelling.
        if fix_first_row:
            for c in range(n):
                self.y[0, c, c].LB = 1.0

        # Row permutations that keep row 0 in place preserve the first row,
        # the clue count and uniqueness, so they combine safely with the digit
        # relabelling above. Canonical form: rows inside each band ascending by
        # their column-0 digit, and bands 1..b-1 ascending by their first row.
        if row_symmetry:

            def col0(r: int) -> gp.LinExpr:
                return gp.quicksum(k * self.y[r, 0, k] for k in range(n))

            for band in range(1 if top_band is not None else 0, b):
                rows = [r for r in range(band * b, (band + 1) * b) if r != 0]
                for r1, r2 in zip(rows, rows[1:]):
                    m.addConstr(col0(r1) + 1 <= col0(r2), name=f"rowsym_{r1}")
            for band in range(1, b - 1):
                m.addConstr(col0(band * b) + 1 <= col0((band + 1) * b), name=f"bandsym_{band}")

        # Objective: clue count, optionally with a positional tie-breaker that
        # prefers clues towards the top-left, weight 2+i+j for cell (i,j).
        # scale exceeds the largest possible tie-breaker sum (at most one clue
        # per cell, weights below 2N), so the clue count is minimised first.
        if position_weights:
            self.scale = 2 * n * n * n + 1
            m.setObjective(
                gp.quicksum(
                    (self.scale + 2 + r + c) * self.x[r, c, k]
                    for r in range(n) for c in range(n) for k in range(n)
                ),
                GRB.MINIMIZE,
            )
        else:
            self.scale = 1
            m.setObjective(self.x.sum(), GRB.MINIMIZE)
        self.delta_keys: set[tuple[int, ...]] = set()

    def clue_bound(self, objective_bound: float) -> Optional[int]:
        """Lower bound on the clue count implied by a bound on the objective."""
        if not math.isfinite(objective_bound) or objective_bound <= -GRB.INFINITY / 2:
            return self.min_clues
        return max(self.min_clues, math.floor(objective_bound / self.scale + 1e-6))

    def _add_spread(self, spread: dict[str, tuple[Optional[int], Optional[int]]]) -> None:
        """Bounds on the number of clues per row, column, box and digit (Ates & Cavdur 2025).

        These are requirements on the puzzle, not valid inequalities: they
        restrict which puzzles are acceptable. The same bounds apply to every
        row (column, box, digit), so they are invariant under the symmetries
        used for symmetry breaking.
        """
        n, m, x = self.n, self.model, self.x
        N = range(n)
        groups = {
            "row": [x.sum(r, "*", "*") for r in N],
            "col": [x.sum("*", c, "*") for c in N],
            "box": [gp.quicksum(x.sum(r, c, "*") for r, c in cells) for cells in box_cells(n)],
            "digit": [x.sum("*", "*", k) for k in N],
        }
        for kind, (lo, hi) in spread.items():
            for i, expr in enumerate(groups[kind]):
                if lo is not None:
                    m.addConstr(expr >= lo, name=f"{kind}_clues_min_{i}")
                if hi is not None:
                    m.addConstr(expr <= hi, name=f"{kind}_clues_max_{i}")

    def _add_strengthening(self, digit_cover: bool, line_pairs: bool, rectangles: str) -> None:
        """Valid inequalities that hold for every grid, added before the first solve."""
        n, m, x, y = self.n, self.model, self.x, self.y
        b = math.isqrt(n)
        N = range(n)

        # Digit cover: if two digits both have no clue, swapping them everywhere
        # gives a second solution. So at least N-1 digits appear among the clues.
        if digit_cover:
            u = m.addVars(n, vtype=GRB.BINARY, name="digit_used")
            m.addConstrs((u[k] <= x.sum("*", "*", k) for k in N), name="digit_used_link")
            m.addConstr(u.sum() >= n - 1, name="digit_cover")

        # Line pairs: two rows of one band (two columns of one stack) can be
        # swapped, so they cannot both be clue-free.
        if line_pairs:
            for r1 in N:
                for r2 in range(r1 + 1, n):
                    if r1 // b == r2 // b:
                        m.addConstr(x.sum(r1, "*", "*") + x.sum(r2, "*", "*") >= 1, name=f"rowpair_{r1}_{r2}")
            for c1 in N:
                for c2 in range(c1 + 1, n):
                    if c1 // b == c2 // b:
                        m.addConstr(x.sum("*", c1, "*") + x.sum("*", c2, "*") >= 1, name=f"colpair_{c1}_{c2}")

        # Rectangles (unavoidable sets of size 4): digits a,b at (r1,c1),(r2,c2)
        # and b,a at (r1,c2),(r2,c1) can be exchanged when the rows share a band
        # or the columns share a stack. If the target contains the pattern, one
        # of its four entries must be a clue.
        added = []
        if rectangles != "off":
            for r1 in N:
                for r2 in range(r1 + 1, n):
                    for c1 in N:
                        for c2 in range(c1 + 1, n):
                            if r1 // b != r2 // b and c1 // b != c2 // b:
                                continue
                            for a in N:
                                for bb in N:
                                    if a == bb:
                                        continue
                                    cells = [(r1, c1, a), (r1, c2, bb), (r2, c1, bb), (r2, c2, a)]
                                    lhs = gp.LinExpr(
                                        [1.0] * 4 + [-1.0] * 4,
                                        [x[i] for i in cells] + [y[i] for i in cells],
                                    )
                                    added.append(m.addLConstr(lhs, GRB.GREATER_EQUAL, -3.0))
            if rectangles == "lazy":
                # Keep them in Gurobi's lazy pool; they enter the LP only when violated.
                m.update()
                m.setAttr("Lazy", added, [3] * len(added))
        self.rectangle_rows = len(added)

    def _contradicts(self, s: np.ndarray) -> gp.LinExpr:
        n = self.n
        return gp.quicksum(
            self.x[r, c, k]
            for r in range(n)
            for c in range(n)
            for k in range(n)
            if k != int(s[r, c])
        )

    def unavoidable_cut(self, target: np.ndarray, s: np.ndarray) -> gp.TempConstr:
        """sum_{(r,c) in D} (x_{r,c,t} - y_{r,c,t}) >= 1 - |D|, D = cells where t != s."""
        n = self.n
        diff = [(r, c) for r in range(n) for c in range(n) if target[r, c] != s[r, c]]
        lhs = gp.quicksum(
            self.x[r, c, int(target[r, c])] - self.y[r, c, int(target[r, c])] for r, c in diff
        )
        return lhs >= 1 - len(diff)

    def add_unavoidable_cut(self, target: np.ndarray, s: np.ndarray) -> bool:
        """Add the unavoidable-set cut to the model unless an existing cut dominates it.

        The cut for key K = {(r, c, t_rc) : (r, c) in D} is implied by the cut
        for any key K' contained in K, because every target that matches the
        pattern K also matches K'. Existing cuts whose key contains K are removed.
        """
        n = self.n
        key = frozenset(
            (r, c, int(target[r, c])) for r in range(n) for c in range(n) if target[r, c] != s[r, c]
        )
        if self.prune_dominated:
            if key in self.unavoidable or any(old <= key for old in self.unavoidable):
                return False
            for old in [old for old in self.unavoidable if key < old]:
                self.model.remove(self.unavoidable.pop(old))
        self.unavoidable[key] = self.model.addConstr(self.unavoidable_cut(target, s))
        return True

    def set_repaired_start(self, target: np.ndarray, clues: np.ndarray, batch: list[np.ndarray]) -> int:
        """MIP start: the last master solution plus clues that rule out every new alternative.

        Adding clues at target values keeps every existing constraint satisfied,
        and a clue in D (where the target differs from s) satisfies both cut
        types for s. Cells are chosen greedily, each covering as many
        not-yet-hit alternatives as possible. Returns the start's clue count.
        """
        n = self.n
        chosen = {(r, c) for r in range(n) for c in range(n) if clues[r, c].any()}
        unhit = []
        for s in batch:
            diff = {(r, c) for r in range(n) for c in range(n) if target[r, c] != s[r, c]}
            if not diff & chosen:
                unhit.append(diff)
        while unhit:
            counts: dict[tuple[int, int], int] = {}
            for diff in unhit:
                for cell in diff:
                    counts[cell] = counts.get(cell, 0) + 1
            best = max(counts, key=counts.get)
            chosen.add(best)
            unhit = [diff for diff in unhit if best not in diff]
        for r in range(n):
            for c in range(n):
                for k in range(n):
                    is_target = int(target[r, c]) == k
                    self.y[r, c, k].Start = float(is_target)
                    self.x[r, c, k].Start = float(is_target and (r, c) in chosen)
        return len(chosen)

    def projected_delta_cuts(self, target: np.ndarray, s: np.ndarray) -> list[gp.TempConstr]:
        """delta_s projected out, one row per cell where the incumbent differs from s."""
        n = self.n
        contra = self._contradicts(s)
        return [
            contra + self.y[r, c, int(s[r, c])] >= 1
            for r in range(n)
            for c in range(n)
            if target[r, c] != s[r, c]
        ]

    def add_delta_cut(self, s: np.ndarray) -> int:
        """Add delta_s, its linking rows and the conditional cut exactly as in the note."""
        key = tuple(int(v) for v in s.ravel())
        if key in self.delta_keys:
            return 0
        self.delta_keys.add(key)
        n, m = self.n, self.model
        delta = m.addVar(lb=0.0, ub=1.0, name=f"delta_{len(self.delta_keys)}")
        # If target differs from s in any cell, delta must equal 1.
        m.addConstrs(delta + self.y[r, c, int(s[r, c])] >= 1 for r in range(n) for c in range(n))
        # If delta=1, at least one selected clue must contradict s.
        m.addConstr(self._contradicts(s) >= delta)
        return 1

    def values(self, source=None) -> tuple[np.ndarray, np.ndarray, int]:
        """Current (target, clues, clue_count) from the model or from a callback."""
        if source is None:
            yv = self.model.getAttr("X", self.y)
            xv = self.model.getAttr("X", self.x)
        else:
            yv = source.cbGetSolution(self.y)
            xv = source.cbGetSolution(self.x)
        return decode(to_array(yv, self.n), to_array(xv, self.n))

    def set_start(self, target: np.ndarray, clues: np.ndarray) -> None:
        for r in range(self.n):
            for c in range(self.n):
                for k in range(self.n):
                    self.y[r, c, k].Start = float(target[r, c] == k)
                    self.x[r, c, k].Start = float(clues[r, c, k])


def generate_start_target(master: Master, env: gp.Env, seed: int,
                          row_symmetry: bool, fixed_top: bool,
                          time_limit: Optional[float]) -> Optional[np.ndarray]:
    """Generate a new grid under the master's fixed digits, then canonicalize rows."""
    if time_limit is not None and time_limit <= 0:
        return None
    n = master.n
    with gp.Model("start_grid", env=env) as model:
        z = model.addVars(n, n, n, vtype=GRB.BINARY)
        add_sudoku_constraints(model, z, n, "start")
        model.Params.Threads = 1
        if time_limit is not None:
            model.Params.TimeLimit = time_limit
        master.model.update()
        rng = random.Random(seed)
        for index, variable in z.items():
            variable.LB = master.y[index].LB
            variable.UB = master.y[index].UB
            variable.Obj = rng.random()
        model.Params.SolutionLimit = 1
        model.optimize()
        if model.SolCount == 0:
            return None
        target = np.argmax(to_array(model.getAttr("X", z), n), axis=2)
    return canonical_rows(target, fixed_top) if row_symmetry else target


def separate(
    sub: Subproblem,
    target: np.ndarray,
    clues: np.ndarray,
    batch_size: int,
    separator: str,
    time_limit: Optional[float],
    deadline: Optional[float] = None,
) -> tuple[Optional[str], list[np.ndarray], bool]:
    """Collect up to batch_size alternatives. Returns (first_status, batch, limited).

    A timeout never establishes uniqueness. Every solve is capped by the
    remaining overall budget, including the first member of a batch.
    """
    batch: list[np.ndarray] = []
    first_status = None
    for sep_it in range(batch_size):
        left = remaining(deadline)
        if left is not None and left <= 0:
            return first_status, batch, True
        limit = time_limit
        if left is not None:
            limit = left if limit is None else min(limit, left)
        status, alt = sub.solve(target, clues, excluded=batch, separator=separator, time_limit=limit)
        if sep_it == 0:
            first_status = status
        if status == "infeasible":
            break
        if status == "limit":
            return first_status, batch, True
        assert alt is not None
        batch.append(alt)
    return first_status, batch, False


def remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.perf_counter())


def run_callback(
    master: Master,
    sub: Subproblem,
    cut: str,
    batch_size: int,
    separator: str,
    subproblem_time_limit: Optional[float],
    verbose: bool,
    start: float,
    deadline: Optional[float],
    recorder: RunRecorder,
) -> dict:
    """Single branch-and-cut: separate every incumbent inside a MIPSOL callback."""
    m = master.model
    m.Params.LazyConstraints = 1
    state = {"cuts": 0, "alternatives": 0, "calls": 0, "limited": False, "history": [], "error": None}

    def callback(model: gp.Model, where: int) -> None:
        try:
            callback_body(model, where)
        except Exception as exc:
            # Gurobi otherwise prints callback exceptions and can keep solving
            # without uniqueness separation. Terminate and propagate instead.
            state["error"] = exc
            model.terminate()

    def callback_body(model: gp.Model, where: int) -> None:
        if where == GRB.Callback.MIP:
            recorder.lower_bound = master.clue_bound(model.cbGet(GRB.Callback.MIP_OBJBND))
            recorder.progress(cuts=state["cuts"], candidates=state["calls"],
                              nodes=model.cbGet(GRB.Callback.MIP_NODCNT))
            if deadline is not None and time.perf_counter() >= deadline:
                model.terminate()
            return
        if where != GRB.Callback.MIPSOL:
            return
        target, clues, clue_count = master.values(source=model)
        first_status, batch, limited = separate(
            sub, target, clues, batch_size, separator, subproblem_time_limit, deadline
        )
        state["calls"] += 1

        for s in batch:
            rows = []
            if cut in ("unavoidable", "both"):
                rows.append(master.unavoidable_cut(target, s))
            if cut in ("delta", "both"):
                rows.extend(master.projected_delta_cuts(target, s))
            for row in rows:
                model.cbLazy(row)
            state["cuts"] += len(rows)
        state["alternatives"] += len(batch)

        bound = master.clue_bound(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
        recorder.lower_bound = bound
        if first_status == "infeasible":
            recorder.consider(target, clues, "Benders incumbent")
        event = {
            "callback": state["calls"],
            "candidate_clues": clue_count,
            "master_bound": bound,
            "new_alternatives": len(batch),
            "unique": first_status == "infeasible",
            "elapsed": time.perf_counter() - start,
        }
        recorder.event(event)
        state["history"].append(event)
        # The complete trace lives in events.jsonl; bound in-memory growth.
        if len(state["history"]) > 2000:
            del state["history"][:1000]
        recorder.progress(cuts=state["cuts"], candidates=state["calls"])
        if verbose:
            verdict = "unique  " if first_status == "infeasible" else f"new={len(batch):2d}"
            print(
                f"candidate {state['calls']:5d}: clues={clue_count:2d}, bound={str(bound):>2}, "
                f"{verdict}, cuts={state['cuts']:6d}, t={time.perf_counter() - start:8.1f}s"
            )

        if limited and not batch:
            # The incumbent could not be verified and no cut was produced.
            # Accepting it silently would be unsound, so stop the solve.
            state["limited"] = True
            model.terminate()

    m.optimize(callback)
    if state["error"] is not None:
        raise state["error"]
    return state


def run_benders(
    n: int = 4,
    mode: str = "callback",
    cut: str = "both",
    batch_size: int = 8,
    separator: str = "auto",
    max_master_solves: int = 100,
    subproblem_time_limit: Optional[float] = None,
    time_limit: Optional[float] = None,
    fix_first_row: bool = True,
    row_symmetry: bool = False,
    digit_cover: bool = False,
    line_pairs: bool = False,
    rectangles: str = "off",
    fixed_grid: Optional[np.ndarray] = None,
    top_band: Optional[np.ndarray] = None,
    speedups: frozenset = frozenset(),
    min_clues: int = 0,
    position_weights: bool = False,
    spread: Optional[dict[str, tuple[Optional[int], Optional[int]]]] = None,
    threads: Optional[int] = None,
    master_params: Optional[list[tuple[str, object]]] = None,
    env_params: Optional[dict[str, object]] = None,
    gurobi_log: bool = False,
    verify: bool = True,
    verbose: bool = True,
    subproblem_threads: Optional[int] = None,
    warm_start: bool = False,
    warm_start_seconds: float = 30.0,
    seed: int = 0,
    run_dir: Optional[Path] = None,
    start_json: Optional[Path] = None,
) -> BendersResult:
    """Run logic-based Benders until uniqueness is proved or a limit is reached."""
    if batch_size < 1 or max_master_solves < 1:
        raise ValueError("batch size and maximum master solves must be positive")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time limit must be positive")
    if subproblem_time_limit is not None and subproblem_time_limit <= 0:
        raise ValueError("subproblem time limit must be positive")
    if warm_start_seconds <= 0 or min_clues < 0:
        raise ValueError("warm start seconds must be positive and minimum clues nonnegative")
    if separator == "auto":
        separator = "closest" if n <= 4 else "feasible"

    start = time.perf_counter()
    deadline = None if time_limit is None else start + time_limit
    version = "{}.{}.{}".format(*gp.gurobi.version())
    recorder = RunRecorder(run_dir, start, deadline,
                           2*n*n*n + 1 if position_weights else 1, verbose)
    recorder.lower_bound = min_clues
    if run_dir is not None:
        atomic_json(run_dir / "environment.json", {
            "platform": platform.platform(), "python_version": platform.python_version(),
            "gurobi_version": version, "numpy_version": np.__version__,
            "size": n, "mode": mode, "cut": cut, "separator": separator,
            "batch_size": batch_size, "time_limit_seconds": time_limit,
            "threads": threads, "subproblem_threads": subproblem_threads,
            "seed": seed, "warm_start": warm_start, "warm_start_seconds": warm_start_seconds,
            "row_symmetry": row_symmetry, "fix_first_row": fix_first_row,
            "digit_cover": digit_cover, "line_pairs": line_pairs, "rectangles": rectangles,
            "min_clues": min_clues, "position_weights": position_weights,
            "speedups": sorted(speedups), "spread": spread or {},
            "fixed_grid": None if fixed_grid is None else fixed_grid.tolist(),
            "top_band": None if top_band is None else top_band.tolist(),
            "master_params": [(name, value) for name, value in (master_params or [])
                              if not any(word in name.upper() for word in ("WLS", "SECRET", "TOKEN", "PASSWORD"))],
        })

    # One environment for both models, so a WLS license uses a single session.
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        for name, value in (env_params or {}).items():
            env.setParam(name, value)
        env.start()

        target_grid = None if fixed_grid is None else complete_grid(n, env, fixed_grid, remaining(deadline))
        master = Master(
            n, env, fix_first_row, row_symmetry,
            digit_cover=digit_cover,
            line_pairs=line_pairs,
            rectangles=rectangles,
            fixed_grid=target_grid,
            top_band=top_band,
            prune_dominated="prune" in speedups,
            min_clues=min_clues,
            position_weights=position_weights,
            spread=spread,
        )
        m = master.model
        m.Params.OutputFlag = 1 if gurobi_log else 0
        m.Params.MIPGap = 0.0  # the clue count is integral; demand a proof
        if threads is not None:
            m.Params.Threads = threads
        for name, value in master_params or []:
            m.setParam(name, value)
        # A caller may tune search, but cannot turn a tolerance stop into an
        # exact optimum certificate by overriding the proof tolerances.
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = 0.0
        sub = Subproblem(n, env, subproblem_time_limit,
                         threads if subproblem_threads is None else subproblem_threads)

        history: list[dict] = []
        target = np.zeros((n, n), dtype=int)
        clues = np.zeros((n, n, n), dtype=np.int8)
        clue_count: Optional[int] = None
        lower_bound: Optional[float] = None
        master_solves = 0
        cuts = 0
        alternatives = 0
        status = "master_iteration_limit"

        if start_json is not None:
            saved = json.loads(start_json.read_text())
            initial_target = np.array(saved["target"], dtype=int) - 1
            initial_puzzle = np.array(saved["puzzle"], dtype=int)
            verdict, _ = find_alternative(initial_target, initial_puzzle, recorder.left())
            if verdict != "infeasible":
                raise ValueError("--start-json puzzle could not be proved unique")
            if np.count_nonzero(initial_puzzle) < min_clues or not meets_spread(initial_puzzle, spread or {}):
                raise ValueError("--start-json puzzle violates clue requirements")
            m.update()
            if any(master.y[r, c, int(initial_target[r, c])].UB < 0.5 or
                   any(master.y[r, c, k].LB > 0.5 and k != initial_target[r, c] for k in range(n))
                   for r in range(n) for c in range(n)):
                raise ValueError("--start-json target conflicts with fixed digits; use matching run settings")
            if row_symmetry and target_grid is None and not np.array_equal(
                    initial_target, canonical_rows(initial_target, top_band is not None)):
                raise ValueError("--start-json target conflicts with row symmetry; use matching run settings")
            initial_clues = np.zeros((n, n, n), dtype=np.int8)
            for r, c in zip(*np.nonzero(initial_puzzle)):
                initial_clues[r, c, int(initial_target[r, c])] = 1
            recorder.consider(initial_target, initial_clues, "saved start", already_checked=True)

        if warm_start and start_json is None:
            warm_started = time.perf_counter()
            left = remaining(deadline)
            budget = warm_start_seconds if left is None else min(warm_start_seconds, left)
            initial_target = target_grid
            if initial_target is None:
                initial_target = generate_start_target(master, env, seed, row_symmetry,
                                                       top_band is not None, min(5.0, budget))
            if initial_target is not None:
                _, witnesses = greedy_start(initial_target, recorder, seed,
                                             max(0.0, budget - (time.perf_counter() - warm_started)),
                                             min_clues, spread or {})
                for witness in witnesses:
                    if cut in ("unavoidable", "both"):
                        cuts += master.add_unavoidable_cut(initial_target, witness)
                    if cut in ("delta", "both"):
                        cuts += master.add_delta_cut(witness)
                alternatives += len(witnesses)

        if recorder.best is not None:
            master.set_start(recorder.best[0], recorder.best[1])
        recorder.progress(force=True, phase="Benders", cuts=cuts)

        if mode == "callback":
            if time_limit is not None:
                m.Params.TimeLimit = remaining(deadline)
            state = run_callback(
                master, sub, cut, batch_size, separator, subproblem_time_limit, verbose, start,
                deadline, recorder,
            )
            master_solves = 1
            cuts += state["cuts"]
            alternatives += state["alternatives"]
            history = state["history"]
            if m.SolCount > 0:
                target, clues, clue_count = master.values()
            if m.Status in MASTER_INFEASIBLE:
                # No puzzle meets the requirements (e.g. --row-clues bounds).
                status = "infeasible"
            else:
                lower_bound = master.clue_bound(m.ObjBound)
                if deadline is not None and time.perf_counter() >= deadline and m.Status != GRB.OPTIMAL:
                    status = "time_limit"
                elif state["limited"]:
                    status = "separation_limit"
                elif m.Status == GRB.OPTIMAL:
                    status = "optimal"
                elif m.Status == GRB.TIME_LIMIT:
                    status = "time_limit"
                else:
                    status = f"gurobi_status_{m.Status}"

        elif mode == "iterative":
            for master_it in range(1, max_master_solves + 1):
                left = remaining(deadline)
                if left is not None:
                    if left <= 0:
                        status = "time_limit"
                        break
                    m.Params.TimeLimit = left
                m.optimize()
                master_solves = master_it

                if m.Status in MASTER_INFEASIBLE:
                    # No puzzle meets the requirements (e.g. --row-clues bounds);
                    # earlier candidates were all cut off, so none is returned.
                    status = "infeasible"
                    clue_count = None
                    break
                if m.Status != GRB.OPTIMAL:
                    # Without an optimal restricted master the bound argument fails.
                    status = "time_limit" if m.Status == GRB.TIME_LIMIT else f"gurobi_status_{m.Status}"
                    if m.SolCount > 0:
                        target, clues, clue_count = master.values()
                    break

                target, clues, clue_count = master.values()
                lower_bound = master.clue_bound(m.ObjBound)
                recorder.lower_bound = lower_bound
                if "bound" in speedups:
                    # Every later restricted master has at least these cuts, so
                    # its optimum cannot be lower: give Gurobi the bound directly.
                    master.lb_constr.RHS = clue_count
                first_status, batch, limited = separate(
                    sub, target, clues, batch_size, separator, subproblem_time_limit, deadline
                )

                event = {
                    "master_iteration": master_it,
                    "master_lower_bound": clue_count,
                    "cuts_before": cuts,
                    "new_alternatives": len(batch),
                    "elapsed": time.perf_counter() - start,
                }
                history.append(event)
                recorder.event(event)
                recorder.progress(cuts=cuts, master_solves=master_solves)
                if verbose:
                    print(
                        f"master {master_it:4d}: clues={clue_count:2d}, cuts={cuts:6d}, "
                        f"new={len(batch):2d}, separator={separator}, "
                        f"t={time.perf_counter() - start:8.1f}s"
                    )

                # If the very first subproblem is infeasible, there is no second
                # completion. The restricted master was solved optimally, so its
                # objective is a lower bound on the full problem: this is optimal.
                if first_status == "infeasible":
                    recorder.consider(target, clues, "Benders incumbent")
                    status = "optimal"
                    break

                for s in batch:
                    if cut in ("unavoidable", "both"):
                        cuts += master.add_unavoidable_cut(target, s)
                    if cut in ("delta", "both"):
                        cuts += master.add_delta_cut(s)
                alternatives += len(batch)
                if "start" in speedups and batch:
                    master.set_repaired_start(target, clues, batch)

                if limited and not batch:
                    status = "time_limit" if remaining(deadline) == 0 else "separation_limit"
                    break
        else:
            raise ValueError("mode must be 'callback' or 'iterative'")

        verified: Optional[bool] = None
        if verify and clue_count is not None:
            # An independent solver catches incumbents accepted despite lazy
            # cuts, and never extends the run with an unbounded final MIP.
            check, _ = find_alternative(target, clue_grid(n, clues), recorder.left())
            verified = None if check == "limit" else check == "infeasible"
            if verified:
                recorder.consider(target, clues, "final incumbent", already_checked=True)
            if status == "optimal" and verified is False:
                # The solver accepted a puzzle that has a second completion,
                # e.g. an incumbent the callback had rejected with lazy cuts.
                # Never report that as optimal.
                status = "invalid_optimal"
            elif status == "optimal" and verified is None:
                status = "verification_limit"

        # A last master candidate need not be unique, especially in iterative
        # mode. Return the best independently verified puzzle found anywhere.
        if recorder.best is not None:
            target, clues, clue_count = recorder.best
            verified = True
            if status == "verification_limit" and lower_bound is not None and clue_count == lower_bound:
                status = "optimal"
        if status != "infeasible":
            lower_bound = max(min_clues, lower_bound if lower_bound is not None else min_clues)
        recorder.lower_bound = lower_bound
        recorder.progress(force=True, status=status, cuts=cuts, master_solves=master_solves)

        return BendersResult(
            status=status,
            size=n,
            mode=mode,
            cut=cut,
            target=target + 1,
            puzzle=clue_grid(n, clues),
            clue_count=clue_count,
            lower_bound=lower_bound,
            master_solves=master_solves,
            subproblem_solves=sub.solves,
            cuts=cuts,
            alternatives=alternatives,
            subproblem_seconds=sub.seconds,
            elapsed_seconds=time.perf_counter() - start,
            verified_unique=verified,
            solver_version=version,
            history=history,
            best_source=recorder.best_source,
            assumed_min_clues=min_clues,
        )


def print_grid(grid: np.ndarray) -> None:
    n = grid.shape[0]
    b = math.isqrt(n)
    width = max(1, len(str(n)))
    horiz = "+" + "+".join(["-" * ((width + 1) * b + 1)] * b) + "+"
    for r in range(n):
        if r % b == 0:
            print(horiz)
        row = []
        for bc in range(b):
            vals = grid[r, bc * b:(bc + 1) * b]
            row.append(" ".join(".".rjust(width) if v == 0 else str(v).rjust(width) for v in vals))
        print("| " + " | ".join(row) + " |")
    print(horiz)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a Sudoku target and a minimum clue set using logic-based "
            "Benders decomposition with Gurobi."
        )
    )
    p.add_argument("--size", "-n", type=int, default=4, choices=[4, 9], help="Sudoku size (4 or 9).")
    p.add_argument(
        "--mode",
        choices=["callback", "iterative"],
        default="callback",
        help="'callback': one branch-and-cut with lazy cuts; 'iterative': re-solve the master loop.",
    )
    p.add_argument(
        "--cut",
        choices=["delta", "unavoidable", "both"],
        default="both",
        help="Benders cut family. 'delta' is the conditional cut of the model note.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Alternatives generated per separation.")
    p.add_argument(
        "--separator",
        choices=["auto", "feasible", "closest", "hybrid"],
        default="auto",
        help="'hybrid': closest search capped at 0.5s, then feasibility if no witness was found.",
    )
    p.add_argument("--max-master-solves", type=int, default=None, help="Iterative mode only.")
    p.add_argument("--time-limit", type=float, default=None, help="Overall time limit in seconds.")
    p.add_argument(
        "--subproblem-time-limit",
        type=float,
        default=None,
        help="Per-subproblem time limit in seconds (optional).",
    )
    p.add_argument("--threads", type=int, default=None, help="Gurobi Threads for master and subproblem.")
    p.add_argument("--subproblem-threads", type=int, default=None, help="Override threads for the small separator model.")
    p.add_argument("--warm-start", action="store_true", help="Generate a grid and greedily remove clues for a verified MIP start.")
    p.add_argument("--warm-start-seconds", type=float, default=30.0, help="Warm start budget, included in the overall time limit.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for generated starts and Gurobi (default 0).")
    p.add_argument("--run-dir", type=Path, default=None, help="Directory for best.json, progress.json and events.jsonl.")
    p.add_argument("--start-json", type=Path, default=None, help="Reuse a verified best.json as a MIP start, with matching model settings.")
    p.add_argument(
        "--gurobi-param",
        type=parse_param,
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Extra Gurobi parameter for the master, e.g. MIPFocus=2. Repeatable.",
    )
    p.add_argument("--gurobi-log", action="store_true", help="Show the Gurobi master log.")
    p.add_argument(
        "--no-symmetry-break",
        action="store_true",
        help="Do not fix the first row to 1..N.",
    )
    p.add_argument(
        "--row-symmetry",
        action="store_true",
        help="Also order rows within bands and bands 2..b by their column-0 digit.",
    )
    p.add_argument(
        "--digit-cover",
        action="store_true",
        help="Require clues for at least N-1 different digits (valid for every grid).",
    )
    p.add_argument(
        "--line-pairs",
        action="store_true",
        help="No two clue-free rows in one band or columns in one stack.",
    )
    p.add_argument(
        "--rectangles",
        choices=["off", "lazy", "model"],
        default="off",
        help="Add the 4-cell unavoidable-set inequalities, in Gurobi's lazy pool or as model rows.",
    )
    p.add_argument(
        "--strengthen",
        action="store_true",
        help="Shorthand for --digit-cover --line-pairs --rectangles lazy.",
    )
    p.add_argument(
        "--fixed-grid",
        default=None,
        metavar="GRID",
        help=(
            "Fix the target grid and only choose clues. GRID is a built-in name "
            f"({', '.join(FIXED_GRIDS)}), a file, or a string of N*N digits; blanks "
            "(0 or .) are completed by a Sudoku solve. Disables symmetry breaking."
        ),
    )
    p.add_argument(
        "--top-band",
        default=None,
        metavar="I|I-J|all",
        help=(
            "Fix the top band to class representative I (from top_bands_<N>.json, made by "
            "make_top_bands.py), or solve one model per class for a range or all classes. "
            "Replaces the first-row symmetry breaking; --time-limit applies per class."
        ),
    )
    for kind, what in (("row", "row"), ("col", "column"), ("box", "box"), ("digit", "digit")):
        p.add_argument(
            f"--{kind}-clues",
            type=parse_range,
            default=None,
            metavar="MIN:MAX",
            help=f"Clues per {what}: MIN:MAX, MIN:, :MAX or an exact number (Ates & Cavdur 2025).",
        )
    p.add_argument(
        "--min-clues",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Assume every unique puzzle has at least M clues and add sum(x) >= M, "
            "e.g. 17 for 9x9 (McGuire et al. 2013). The clue count is still minimised."
        ),
    )
    p.add_argument(
        "--position-weights",
        action="store_true",
        help=(
            "Tie-breaker: among puzzles with the fewest clues, minimise sum (2+i+j) x_ijk, "
            "i.e. prefer clues towards the top-left."
        ),
    )
    p.add_argument(
        "--speedups",
        type=parse_speedups,
        default=frozenset(),
        metavar="LIST",
        help=(
            "Iterative mode: comma-separated subset of bound (master lower-bound row), "
            "start (repaired MIP start) and prune (drop dominated unavoidable-set cuts), "
            "or 'none'. Default: none (none of them helped in our tests)."
        ),
    )
    p.add_argument(
        "--license-file",
        type=Path,
        default=None,
        help="Gurobi license file to use (sets GRB_LICENSE_FILE), e.g. a version-14 license.",
    )
    p.add_argument(
        "--wls-file",
        type=Path,
        default=None,
        help=(
            "File with WLSACCESSID=, WLSSECRET= and LICENSEID= lines. The values are "
            "passed to gp.Env as parameters (Web License Service)."
        ),
    )
    p.add_argument("--no-verify", action="store_true", help="Skip the final uniqueness re-check.")
    p.add_argument("--output-json", type=Path, default=None, help="Optional JSON result file.")
    p.add_argument("--quiet", action="store_true", help="Suppress iteration log.")
    return p


def load_top_bands(n: int) -> list[tuple[str, int]]:
    """Band class representatives written by make_top_bands.py."""
    path = Path(__file__).with_name(f"top_bands_{n}.json")
    if not path.is_file():
        raise ValueError(f"{path.name} not found; run make_top_bands.py --size {n}")
    return [(text, int(size)) for text, size in json.loads(path.read_text())]


def select_top_bands(spec: str, n: int) -> list[tuple[int, str, int]]:
    """Parse 'all', 'I' or 'I-J' (inclusive) into (index, band, class size) triples."""
    bands = load_top_bands(n)
    if spec == "all":
        lo, hi = 0, len(bands) - 1
    elif "-" in spec:
        lo, hi = (int(part) for part in spec.split("-", 1))
    else:
        lo = hi = int(spec)
    if not 0 <= lo <= hi < len(bands):
        raise ValueError(f"{spec!r} is outside 0..{len(bands) - 1}")
    return [(i, *bands[i]) for i in range(lo, hi + 1)]


def band_array(text: str, n: int) -> np.ndarray:
    """Band string of digits 1..N to a (sqrt(N), N) array of 0-based digits."""
    return np.array([int(ch) - 1 for ch in text]).reshape(math.isqrt(n), n)


def print_options(args: argparse.Namespace) -> None:
    print(
        f"Options: digit_cover={args.digit_cover}, line_pairs={args.line_pairs}, "
        f"rectangles={args.rectangles}, row_symmetry={args.row_symmetry}, "
        f"speedups={','.join(sorted(args.speedups)) or 'none'}, fixed_grid={args.fixed_grid}, "
        f"top_band={args.top_band}, min_clues={args.min_clues}, "
        f"position_weights={args.position_weights}, row/col/box/digit clues="
        f"{args.row_clues}/{args.col_clues}/{args.box_clues}/{args.digit_clues}"
    )


def run_band_split(args: argparse.Namespace, common: dict, top_bands: list[tuple[int, str, int]]) -> int:
    """Solve one model per top-band class; the problem optimum is the minimum over classes.

    The overall lower bound is the smallest per-class lower bound, and the
    best upper bound is the smallest verified unique puzzle over all classes.
    --time-limit applies to each class separately.
    """
    print_options(args)
    print(f"Solving {len(top_bands)} top-band classes, time limit per class: {args.time_limit}")
    print(f"{'class':>5} {'size':>5} {'status':>22} {'clues':>5} {'LB':>5} {'unique':>6} {'time':>8}")
    results = []
    common = dict(common, verbose=False)
    for index, text, size in top_bands:
        settings = dict(common)
        if settings.get("run_dir") is not None:
            settings["run_dir"] = settings["run_dir"] / f"class-{index:03d}"
        result = run_benders(**settings, top_band=band_array(text, args.size))
        results.append((index, text, size, result))
        lb = result.lower_bound if result.lower_bound is not None else 0.0
        print(
            f"{index:5d} {size:5d} {result.status:>22} {str(result.clue_count):>5} "
            f"{lb:5.1f} {str(result.verified_unique):>6} {result.elapsed_seconds:8.1f}",
            flush=True,
        )

    # Classes without any puzzle meeting the requirements do not bound the minimum.
    open_classes = [r for *_, r in results if r.status != "infeasible"]
    if not open_classes:
        print("\nNo top-band class admits a puzzle meeting the requirements.")
        return 2
    lower = min(math.ceil((r.lower_bound or 0.0) - 1e-6) for r in open_classes)
    unique = [(r.clue_count, i, r) for i, _, _, r in results if r.verified_unique]
    print(f"\nLower bound over all classes: {lower}")
    if unique:
        best_count, best_index, best = min(unique, key=lambda item: (item[0], item[1]))
        print(f"Best verified unique puzzle: {best_count} clues (class {best_index})")
        print("Proven optimal:", best_count == lower)
        print("\nTarget Sudoku:")
        print_grid(best.target)
        print("\nPuzzle (. = blank):")
        print_grid(best.puzzle)
    else:
        best_count = None
        print("No verified unique puzzle found.")

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(
            [dict(r.as_jsonable(), top_band_class=i, top_band=t, class_size=s) for i, t, s, r in results],
            indent=2,
        ))
        print(f"\nWrote {args.output_json}")
    return 0 if best_count is not None and best_count == lower else 2


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not any(name.lower() == "seed" for name, _ in args.gurobi_param):
        args.gurobi_param.append(("Seed", args.seed))

    # Conservative defaults: 4x4 is a teaching/test case; 9x9 is much harder.
    if args.batch_size is None:
        args.batch_size = 8 if args.size == 4 else 4
    if args.max_master_solves is None:
        args.max_master_solves = 100 if args.size == 4 else 500

    if args.license_file is not None:
        # Read by Gurobi when each environment starts, so setting it here is enough.
        os.environ["GRB_LICENSE_FILE"] = str(args.license_file.expanduser())

    if args.strengthen:
        args.digit_cover = args.line_pairs = True
        if args.rectangles == "off":
            args.rectangles = "lazy"

    fixed_grid = None
    if args.fixed_grid is not None:
        try:
            fixed_grid = parse_grid(args.fixed_grid, args.size)
        except (OSError, ValueError) as exc:
            print(f"Cannot use --fixed-grid: {exc}", file=sys.stderr)
            return 3

    env_params = None
    if args.wls_file is not None:
        try:
            env_params = read_wls_file(args.wls_file)
        except (OSError, ValueError) as exc:
            print(f"Cannot read WLS credentials: {exc}", file=sys.stderr)
            return 3

    top_bands: list[tuple[int, str, int]] = []
    if args.top_band is not None:
        try:
            if fixed_grid is not None:
                raise ValueError("cannot be combined with --fixed-grid")
            top_bands = select_top_bands(args.top_band, args.size)
        except (OSError, ValueError) as exc:
            print(f"Cannot use --top-band: {exc}", file=sys.stderr)
            return 3

    common = dict(
        n=args.size,
        mode=args.mode,
        cut=args.cut,
        batch_size=args.batch_size,
        separator=args.separator,
        max_master_solves=args.max_master_solves,
        subproblem_time_limit=args.subproblem_time_limit,
        time_limit=args.time_limit,
        fix_first_row=not args.no_symmetry_break,
        row_symmetry=args.row_symmetry,
        digit_cover=args.digit_cover,
        line_pairs=args.line_pairs,
        rectangles=args.rectangles,
        fixed_grid=fixed_grid,
        speedups=args.speedups,
        min_clues=args.min_clues,
        position_weights=args.position_weights,
        spread={
            kind: getattr(args, f"{kind}_clues")
            for kind in ("row", "col", "box", "digit")
            if getattr(args, f"{kind}_clues") is not None
        },
        threads=args.threads,
        master_params=args.gurobi_param,
        env_params=env_params,
        gurobi_log=args.gurobi_log,
        verify=not args.no_verify,
        verbose=not args.quiet,
        subproblem_threads=args.subproblem_threads,
        warm_start=args.warm_start,
        warm_start_seconds=args.warm_start_seconds,
        seed=args.seed,
        run_dir=args.run_dir,
        start_json=args.start_json,
    )

    try:
        if len(top_bands) > 1:
            return run_band_split(args, common, top_bands)
        band = band_array(top_bands[0][1], args.size) if top_bands else None
        result = run_benders(**common, top_band=band)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Cannot run: {exc}", file=sys.stderr)
        return 3
    except gp.GurobiError as exc:
        print(f"Gurobi error: {exc}", file=sys.stderr)
        if "license" in str(exc).lower():
            version = "{}.{}.{}".format(*gp.gurobi.version())
            source = (
                f"WLS credentials from {args.wls_file}"
                if args.wls_file is not None
                else f"the license {os.environ.get('GRB_LICENSE_FILE', '(default location)')}"
            )
            print(
                f"gurobipy {version} could not use {source}. A license valid for this "
                "major version is needed; pass it with --license-file or --wls-file.",
                file=sys.stderr,
            )
        return 3

    print("\nGurobi version:", result.solver_version)
    print("Mode / cut:", result.mode, "/", result.cut)
    print_options(args)
    if top_bands:
        index, text, size = top_bands[0]
        print(f"Top band: class {index} ({text}, {size} bands)")
    print("Status:", result.status)
    print("Clue count:", result.clue_count)
    print("Lower bound:", result.lower_bound, f"(assumes >= {args.min_clues})" if args.min_clues else "")
    print("Master solves:", result.master_solves)
    print("Subproblem solves:", result.subproblem_solves)
    print("Alternatives found:", result.alternatives)
    print("Benders cuts added:", result.cuts)
    print("Verified unique:", result.verified_unique)
    print(f"Subproblem time: {result.subproblem_seconds:.3f} s")
    print(f"Elapsed: {result.elapsed_seconds:.3f} s")

    if result.clue_count is not None:
        print("\nGenerated target Sudoku:")
        print_grid(result.target)
        print("\nGenerated puzzle (. = blank):")
        print_grid(result.puzzle)

    if args.output_json is not None:
        atomic_json(args.output_json, result.as_jsonable())
        print(f"\nWrote {args.output_json}")

    return 0 if result.status == "optimal" else 2


if __name__ == "__main__":
    sys.exit(main())
