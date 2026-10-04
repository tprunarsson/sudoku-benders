#!/usr/bin/env python3
"""Generate a Sudoku and a minimum clue set with logic-based Benders decomposition.

The master chooses both:
  y[r,c,k] = 1 if digit k is the generated target value in cell (r,c)
  x[r,c,k] = 1 if that target value is exposed as a clue

The subproblem searches for a second complete Sudoku satisfying the selected clues.
If such a solution s exists, a conditional Benders cut is added. The cut says:
if the master does not choose s itself as the target, then at least one selected
clue must contradict s.

The MILPs are solved by scipy.optimize.milp, whose backend is HiGHS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix


@dataclass
class BendersResult:
    status: str
    size: int
    target: np.ndarray
    puzzle: np.ndarray
    clue_count: int
    master_solves: int
    subproblem_solves: int
    cuts: int
    elapsed_seconds: float
    history: list[dict]

    def as_jsonable(self) -> dict:
        return {
            "status": self.status,
            "size": self.size,
            "target": self.target.tolist(),
            "puzzle": self.puzzle.tolist(),
            "clue_count": self.clue_count,
            "master_solves": self.master_solves,
            "subproblem_solves": self.subproblem_solves,
            "cuts": self.cuts,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history,
        }


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


def vidx(n: int, r: int, c: int, k: int) -> int:
    """Flat index of one-hot variable for cell (r,c), digit k."""
    return (r * n + c) * n + k


def sudoku_rows(n: int) -> list[tuple[dict[int, float], float, float]]:
    """Return row descriptions for the standard Sudoku constraints."""
    rows: list[tuple[dict[int, float], float, float]] = []

    # Exactly one digit in each cell.
    for r in range(n):
        for c in range(n):
            rows.append(({vidx(n, r, c, k): 1.0 for k in range(n)}, 1.0, 1.0))

    # Each digit exactly once in each row.
    for r in range(n):
        for k in range(n):
            rows.append(({vidx(n, r, c, k): 1.0 for c in range(n)}, 1.0, 1.0))

    # Each digit exactly once in each column.
    for c in range(n):
        for k in range(n):
            rows.append(({vidx(n, r, c, k): 1.0 for r in range(n)}, 1.0, 1.0))

    # Each digit exactly once in each box.
    for cells in box_cells(n):
        for k in range(n):
            rows.append((
                {vidx(n, r, c, k): 1.0 for r, c in cells},
                1.0,
                1.0,
            ))

    return rows


def make_constraint(
    rows: Iterable[tuple[dict[int, float], float, float]], nvars: int
) -> LinearConstraint:
    rows = list(rows)
    a = lil_matrix((len(rows), nvars), dtype=float)
    lb = np.empty(len(rows), dtype=float)
    ub = np.empty(len(rows), dtype=float)

    for i, (coeff, lo, hi) in enumerate(rows):
        for j, value in coeff.items():
            a[i, j] = value
        lb[i] = lo
        ub[i] = hi

    return LinearConstraint(csr_matrix(a), lb, ub)


def solve_master(
    n: int,
    discovered_solutions: list[np.ndarray],
    fix_first_row: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the restricted master problem.

    For every discovered complete Sudoku s, introduce delta_s in [0,1].
    Linking constraints force delta_s=1 whenever the current target differs
    from s. Then the cut requires at least one clue that s violates.
    """
    n3 = n ** 3
    m = len(discovered_solutions)

    y0 = 0
    x0 = n3
    d0 = 2 * n3
    nvars = 2 * n3 + m

    rows = sudoku_rows(n)

    # x_ijk <= y_ijk: a clue can reveal only the generated target value.
    for r in range(n):
        for c in range(n):
            for k in range(n):
                rows.append((
                    {
                        x0 + vidx(n, r, c, k): 1.0,
                        y0 + vidx(n, r, c, k): -1.0,
                    },
                    -np.inf,
                    0.0,
                ))

    # Safe symmetry breaking under digit relabelling.
    if fix_first_row:
        for c in range(n):
            rows.append(({y0 + vidx(n, 0, c, c): 1.0}, 1.0, 1.0))

    # Conditional logic-based Benders cuts.
    for cut_id, s in enumerate(discovered_solutions):
        delta = d0 + cut_id

        # If target differs from s in any cell, delta must equal 1.
        # delta + y_{r,c,s_rc} >= 1 for all cells.
        for r in range(n):
            for c in range(n):
                rows.append((
                    {
                        delta: 1.0,
                        y0 + vidx(n, r, c, int(s[r, c])): 1.0,
                    },
                    1.0,
                    np.inf,
                ))

        # If delta=1, at least one selected clue must contradict s.
        # sum_{r,c,k != s_rc} x_rck >= delta.
        coeff: dict[int, float] = {delta: -1.0}
        for r in range(n):
            for c in range(n):
                sk = int(s[r, c])
                for k in range(n):
                    if k != sk:
                        coeff[x0 + vidx(n, r, c, k)] = 1.0
        rows.append((coeff, 0.0, np.inf))

    c = np.zeros(nvars)
    c[x0:x0 + n3] = 1.0

    # y and x must be binary. delta can remain continuous in [0,1]:
    # integral y plus the linking constraints force delta to 0 or 1 when needed.
    integrality = np.zeros(nvars)
    integrality[:2 * n3] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(np.zeros(nvars), np.ones(nvars)),
        constraints=make_constraint(rows, nvars),
        options={"disp": False},
    )

    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"Master failed: status={result.status}; message={result.message}"
        )

    y = result.x[y0:y0 + n3].reshape(n, n, n)
    x = result.x[x0:x0 + n3].reshape(n, n, n)
    target = np.argmax(y, axis=2)
    clues = (x > 0.5).astype(np.int8)
    return target, clues, int(round(result.fun))


def solve_subproblem(
    n: int,
    target: np.ndarray,
    clues: np.ndarray,
    excluded: Optional[list[np.ndarray]] = None,
    separator: str = "feasible",
    time_limit: Optional[float] = None,
) -> tuple[str, Optional[np.ndarray]]:
    """Find a second Sudoku satisfying the current clues.

    Status is one of:
      alternative : a second complete Sudoku was found
      infeasible  : no second Sudoku exists (uniqueness certificate)
      limit       : solver stopped without a conclusion
    """
    excluded = [] if excluded is None else excluded
    n3 = n ** 3
    rows = sudoku_rows(n)

    # Enforce selected clues.
    for r in range(n):
        for c in range(n):
            for k in range(n):
                if clues[r, c, k] > 0:
                    rows.append(({vidx(n, r, c, k): 1.0}, 1.0, 1.0))

    # Require a solution different from the current generated target.
    rows.append((
        {
            vidx(n, r, c, int(target[r, c])): 1.0
            for r in range(n)
            for c in range(n)
        },
        -np.inf,
        n * n - 1,
    ))

    # During multi-cut separation, exclude alternatives already returned
    # in the current batch so HiGHS finds a new witness.
    for s in excluded:
        rows.append((
            {
                vidx(n, r, c, int(s[r, c])): 1.0
                for r in range(n)
                for c in range(n)
            },
            -np.inf,
            n * n - 1,
        ))

    objective = np.zeros(n3)
    if separator == "closest":
        # Maximize agreement with target, equivalently minimize Hamming distance.
        for r in range(n):
            for c in range(n):
                objective[vidx(n, r, c, int(target[r, c]))] = -1.0
    elif separator != "feasible":
        raise ValueError("separator must be 'feasible' or 'closest'")

    options: dict[str, object] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    result = milp(
        c=objective,
        integrality=np.ones(n3),
        bounds=Bounds(np.zeros(n3), np.ones(n3)),
        constraints=make_constraint(rows, n3),
        options=options,
    )

    # HiGHS/SciPy status 2 = infeasible. This is the uniqueness certificate.
    if result.status == 2:
        return "infeasible", None

    # If a feasible incumbent exists, it is enough to make a valid cut,
    # even if a closest-solution optimization hit its time limit.
    if result.x is not None:
        z = result.x.reshape(n, n, n)
        return "alternative", np.argmax(z, axis=2)

    return "limit", None


def clue_grid(n: int, clues: np.ndarray) -> np.ndarray:
    """Return an N x N puzzle with 0 for blanks and 1..N for clues."""
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            ks = np.where(clues[r, c, :] > 0)[0]
            if len(ks):
                out[r, c] = int(ks[0]) + 1
    return out


def run_benders(
    n: int = 4,
    batch_size: int = 8,
    separator: str = "auto",
    max_master_solves: int = 100,
    subproblem_time_limit: Optional[float] = None,
    fix_first_row: bool = True,
    verbose: bool = True,
) -> BendersResult:
    """Run logic-based Benders until uniqueness is proved or a limit is reached."""
    if separator == "auto":
        separator = "closest" if n <= 4 else "feasible"

    discovered: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    history: list[dict] = []
    subproblem_solves = 0
    start = time.perf_counter()

    target = np.zeros((n, n), dtype=int)
    clues = np.zeros((n, n, n), dtype=np.int8)
    clue_count = 0

    for master_it in range(1, max_master_solves + 1):
        target, clues, clue_count = solve_master(
            n=n,
            discovered_solutions=discovered,
            fix_first_row=fix_first_row,
        )

        batch: list[np.ndarray] = []
        first_status = None
        separation_limited = False

        for sep_it in range(batch_size):
            status, alt = solve_subproblem(
                n=n,
                target=target,
                clues=clues,
                excluded=batch,
                separator=separator,
                time_limit=subproblem_time_limit,
            )
            subproblem_solves += 1
            if sep_it == 0:
                first_status = status

            if status == "infeasible":
                break
            if status == "limit":
                separation_limited = True
                break

            assert alt is not None
            key = tuple(int(v) for v in alt.ravel())
            if key in seen:
                # It is already globally represented by a master cut.
                break
            seen.add(key)
            batch.append(alt)

        history.append({
            "master_iteration": master_it,
            "master_lower_bound": clue_count,
            "cuts_before": len(discovered),
            "new_cuts": len(batch),
        })

        if verbose:
            print(
                f"master {master_it:3d}: clues={clue_count:2d}, "
                f"cuts={len(discovered):4d}, new={len(batch):2d}, "
                f"separator={separator}"
            )

        # If the very first subproblem is infeasible, there is no second
        # completion for this puzzle. Since the restricted master was solved
        # optimally, its objective is also a lower bound on the full problem;
        # therefore this unique puzzle is globally optimal for the model.
        if first_status == "infeasible":
            elapsed = time.perf_counter() - start
            return BendersResult(
                status="optimal",
                size=n,
                target=target + 1,
                puzzle=clue_grid(n, clues),
                clue_count=clue_count,
                master_solves=master_it,
                subproblem_solves=subproblem_solves,
                cuts=len(discovered),
                elapsed_seconds=elapsed,
                history=history,
            )

        discovered.extend(batch)

        if separation_limited and not batch:
            elapsed = time.perf_counter() - start
            return BendersResult(
                status="separation_limit",
                size=n,
                target=target + 1,
                puzzle=clue_grid(n, clues),
                clue_count=clue_count,
                master_solves=master_it,
                subproblem_solves=subproblem_solves,
                cuts=len(discovered),
                elapsed_seconds=elapsed,
                history=history,
            )

        if not batch:
            # Defensive termination: no new information was generated.
            elapsed = time.perf_counter() - start
            return BendersResult(
                status="stalled",
                size=n,
                target=target + 1,
                puzzle=clue_grid(n, clues),
                clue_count=clue_count,
                master_solves=master_it,
                subproblem_solves=subproblem_solves,
                cuts=len(discovered),
                elapsed_seconds=elapsed,
                history=history,
            )

    elapsed = time.perf_counter() - start
    return BendersResult(
        status="master_iteration_limit",
        size=n,
        target=target + 1,
        puzzle=clue_grid(n, clues),
        clue_count=clue_count,
        master_solves=max_master_solves,
        subproblem_solves=subproblem_solves,
        cuts=len(discovered),
        elapsed_seconds=elapsed,
        history=history,
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
            "Benders decomposition with SciPy/HiGHS."
        )
    )
    p.add_argument("--size", "-n", type=int, default=4, choices=[4, 9], help="Sudoku size (4 or 9).")
    p.add_argument("--batch-size", type=int, default=None, help="Alternatives generated per master solve.")
    p.add_argument(
        "--separator",
        choices=["auto", "feasible", "closest"],
        default="auto",
        help="Subproblem objective. 'closest' is stronger but can be expensive for 9x9.",
    )
    p.add_argument("--max-master-solves", type=int, default=None, help="Maximum restricted-master solves.")
    p.add_argument(
        "--subproblem-time-limit",
        type=float,
        default=None,
        help="Per-subproblem HiGHS time limit in seconds (optional).",
    )
    p.add_argument(
        "--no-symmetry-break",
        action="store_true",
        help="Do not fix the first row to 1..N.",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Optional JSON result file.")
    p.add_argument("--quiet", action="store_true", help="Suppress iteration log.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Conservative defaults: 4x4 is a teaching/test case; 9x9 is much harder.
    if args.batch_size is None:
        args.batch_size = 8 if args.size == 4 else 4
    if args.max_master_solves is None:
        args.max_master_solves = 100 if args.size == 4 else 500

    result = run_benders(
        n=args.size,
        batch_size=args.batch_size,
        separator=args.separator,
        max_master_solves=args.max_master_solves,
        subproblem_time_limit=args.subproblem_time_limit,
        fix_first_row=not args.no_symmetry_break,
        verbose=not args.quiet,
    )

    print("\nStatus:", result.status)
    print("Clue count:", result.clue_count)
    print("Master solves:", result.master_solves)
    print("Subproblem solves:", result.subproblem_solves)
    print("Stored Benders cuts:", result.cuts)
    print(f"Elapsed: {result.elapsed_seconds:.3f} s")

    print("\nGenerated target Sudoku:")
    print_grid(result.target)
    print("\nGenerated puzzle (. = blank):")
    print_grid(result.puzzle)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result.as_jsonable(), indent=2))
        print(f"\nWrote {args.output_json}")

    return 0 if result.status == "optimal" else 2


if __name__ == "__main__":
    sys.exit(main())
