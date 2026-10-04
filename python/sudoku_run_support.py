"""Warm starts and durable progress for long Sudoku Benders runs."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from sudoku_search import find_alternative


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


class RunRecorder:
    def __init__(self, directory: Path | None, start: float, deadline: float | None,
                 scale: int = 1, verbose: bool = True):
        self.directory, self.start, self.deadline = directory, start, deadline
        self.scale, self.verbose = scale, verbose
        self.best: tuple[np.ndarray, np.ndarray, int] | None = None
        self.best_score: int | None = None
        self.lower_bound: int | None = None
        self.last_progress = -math.inf
        self.best_source: str | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def left(self) -> float | None:
        return None if self.deadline is None else max(0.0, self.deadline - time.perf_counter())

    def consider(self, target: np.ndarray, clues: np.ndarray, source: str,
                 already_checked: bool = False) -> bool:
        count = int(clues.sum())
        secondary = sum((2+r+c) * int(clues[r, c].sum())
                        for r in range(len(target)) for c in range(len(target)))
        score = count * self.scale + (secondary if self.scale > 1 else 0)
        if self.best_score is not None and score >= self.best_score:
            return False
        puzzle = (target + 1) * clues.sum(axis=2)
        if not already_checked:
            left = self.left()
            verdict, _ = find_alternative(target, puzzle, min(5.0, left) if left is not None else 5.0)
            if verdict != "infeasible":
                return False
        self.best = (target.copy(), clues.copy(), count)
        self.best_score, self.best_source = score, source
        if self.directory is not None:
            atomic_json(self.directory / "best.json", {
                "target": (target + 1).tolist(), "puzzle": puzzle.tolist(),
                "clue_count": count, "verified_unique": True,
                "verification": "independent exact backtracking", "source": source,
                "elapsed_seconds": time.perf_counter() - self.start,
            })
        if self.verbose:
            print(f"Best verified puzzle: {count} clues ({source}), "
                  f"t={time.perf_counter() - self.start:.1f}s", flush=True)
        self.progress(force=True)
        return True

    def event(self, event: dict) -> None:
        if self.directory is not None:
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")

    def progress(self, force: bool = False, **data) -> None:
        now = time.perf_counter()
        if not force and now - self.last_progress < 30.0:
            return
        self.last_progress = now
        if self.directory is not None:
            atomic_json(self.directory / "progress.json", {
                "elapsed_seconds": now - self.start,
                "best_clue_count": None if self.best is None else self.best[2],
                "lower_bound": self.lower_bound, **data,
            })


def canonical_rows(target: np.ndarray, fixed_top: bool) -> np.ndarray:
    """Order rows/bands without moving the first row or a fixed top band."""
    b = math.isqrt(len(target))
    rows = target.copy()
    for band in range(1 if fixed_top else 0, b):
        indices = list(range(band*b + (1 if band == 0 else 0), (band+1)*b))
        rows[indices] = rows[sorted(indices, key=lambda r: int(rows[r, 0]))]
    bands = sorted(range(1, b), key=lambda band: int(rows[band*b, 0]))
    return np.concatenate([rows[:b]] + [rows[band*b:(band+1)*b] for band in bands])


def meets_spread(puzzle: np.ndarray, spread: dict, lower_only: bool = False) -> bool:
    n = len(puzzle)
    b = math.isqrt(n)
    shown = puzzle != 0
    groups = {
        "row": shown.sum(axis=1), "col": shown.sum(axis=0),
        "box": [shown[r:r+b, c:c+b].sum() for r in range(0, n, b) for c in range(0, n, b)],
        "digit": [np.count_nonzero(puzzle == k) for k in range(1, n+1)],
    }
    return all(all((lo is None or count >= lo) and
                   (lower_only or hi is None or count <= hi) for count in groups[kind])
               for kind, (lo, hi) in spread.items())


def greedy_start(target: np.ndarray, recorder: RunRecorder, seed: int,
                 budget: float, min_clues: int, spread: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """Remove clues only after an exact uniqueness proof; retain witnesses as cuts."""
    n = len(target)
    puzzle = target + 1
    deadline = time.perf_counter() + budget
    if recorder.deadline is not None:
        deadline = min(deadline, recorder.deadline)
    alternatives = []
    cells = list(range(n*n))
    random.Random(seed).shuffle(cells)

    def clues_for(grid):
        clues = np.zeros((n, n, n), dtype=np.int8)
        for r, c in zip(*np.nonzero(grid)):
            clues[r, c, int(target[r, c])] = 1
        return clues

    if np.count_nonzero(puzzle) >= min_clues and meets_spread(puzzle, spread):
        recorder.consider(target, clues_for(puzzle), "generated start", already_checked=True)
    for cell in cells:
        left = deadline - time.perf_counter()
        if left <= 0 or np.count_nonzero(puzzle) <= min_clues:
            break
        r, c = divmod(cell, n)
        digit = int(puzzle[r, c])
        puzzle[r, c] = 0
        if not meets_spread(puzzle, spread, lower_only=True):
            puzzle[r, c] = digit
            continue
        verdict, alternative = find_alternative(target, puzzle, min(0.25, left))
        if verdict != "infeasible":
            puzzle[r, c] = digit
            if alternative is not None:
                alternatives.append(alternative)
        elif meets_spread(puzzle, spread):
            recorder.consider(target, clues_for(puzzle), "greedy start", already_checked=True)
    return clues_for(puzzle), alternatives
