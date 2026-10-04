"""Independent exact Sudoku checker used for starts and result verification.

The MIP separator still generates the Benders cuts. This small bit-mask search
also checks results without depending on the MIP solver's lazy-cut handling.
Digits are zero-based in targets and one-based in puzzles (zero is blank).
"""

from __future__ import annotations

import math
import time

import numpy as np


def find_alternative(target: np.ndarray, puzzle: np.ndarray,
                     time_limit: float | None = None) -> tuple[str, np.ndarray | None]:
    """Return alternative, infeasible (unique), or limit; never infer uniqueness on timeout."""
    n = len(target)
    b = math.isqrt(n)
    if b * b != n or target.shape != (n, n) or puzzle.shape != (n, n):
        raise ValueError("expected square Sudoku arrays with square boxes")
    if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(puzzle.dtype, np.integer):
        raise ValueError("Sudoku digits must be integers")
    if np.any(target < 0) or np.any(target >= n) or np.any(puzzle < 0) or np.any(puzzle > n):
        raise ValueError("Sudoku digits out of range")
    wanted = set(range(n))
    units = list(target) + list(target.T)
    units += [target[r:r+b, c:c+b].ravel() for r in range(0, n, b) for c in range(0, n, b)]
    if any(set(unit) != wanted for unit in units):
        raise ValueError("target is not a valid completed Sudoku")
    if np.any((puzzle != 0) & (puzzle != target + 1)):
        raise ValueError("puzzle clues disagree with target")

    deadline = None if time_limit is None else time.perf_counter() + max(0.0, time_limit)
    full = (1 << n) - 1
    rows, cols, boxes = [0] * n, [0] * n, [0] * n
    grid = (puzzle - 1).ravel().tolist()
    expected = target.ravel().tolist()
    positions = [(i // n, i % n, (i // n // b) * b + (i % n // b)) for i in range(n*n)]
    empty = []
    for i, value in enumerate(grid):
        if value < 0:
            empty.append(i)
        else:
            r, c, box = positions[i]
            bit = 1 << value
            rows[r] |= bit
            cols[c] |= bit
            boxes[box] |= bit

    def search(depth: int, different: bool) -> list[int] | None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError
        if depth == len(empty):
            return grid.copy() if different else None
        best, mask, size = depth, 0, n + 1
        for j in range(depth, len(empty)):
            r, c, box = positions[empty[j]]
            available = full & ~(rows[r] | cols[c] | boxes[box])
            count = available.bit_count()
            if count == 0:
                return None
            if count < size:
                best, mask, size = j, available, count
                if count == 1:
                    break
        empty[depth], empty[best] = empty[best], empty[depth]
        i = empty[depth]
        r, c, box = positions[i]
        # Try disagreement first to return witnesses quickly on sparse puzzles.
        expected_bit = 1 << expected[i]
        bits = mask & ~expected_bit
        choices = []
        while bits:
            bit = bits & -bits
            choices.append(bit)
            bits ^= bit
        if mask & expected_bit:
            choices.append(expected_bit)
        for bit in choices:
            value = bit.bit_length() - 1
            grid[i] = value
            rows[r] |= bit
            cols[c] |= bit
            boxes[box] |= bit
            result = search(depth + 1, different or value != expected[i])
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[box] ^= bit
            if result is not None:
                return result
        grid[i] = -1
        empty[depth], empty[best] = empty[best], empty[depth]
        return None

    try:
        result = search(0, False)
    except TimeoutError:
        return "limit", None
    if result is None:
        return "infeasible", None
    return "alternative", np.array(result, dtype=int).reshape(n, n)
