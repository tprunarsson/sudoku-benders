import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sudoku_search import find_alternative


TARGET = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]])


@pytest.fixture(scope="module")
def all_grids():
    permutations = list(itertools.permutations(range(4)))
    grids = []
    for rows in itertools.product(permutations, repeat=4):
        if any(len({rows[r][c] for r in range(4)}) != 4 for c in range(4)):
            continue
        if any(len({rows[r][c] for r in range(br, br+2) for c in range(bc, bc+2)}) != 4
               for br in (0, 2) for bc in (0, 2)):
            continue
        grids.append(rows)
    assert len(grids) == 288
    return np.array(grids)


def test_checker_agrees_with_complete_enumeration(all_grids):
    rng = np.random.default_rng(73)
    for clues in range(17):
        for _ in range(8):
            shown = np.zeros(16, dtype=bool)
            shown[rng.choice(16, clues, replace=False)] = True
            shown = shown.reshape(4, 4)
            puzzle = np.where(shown, TARGET + 1, 0)
            matching = all_grids[np.all((all_grids == TARGET) | ~shown, axis=(1, 2))]
            verdict, alternative = find_alternative(TARGET, puzzle)
            assert (verdict == "infeasible") == (len(matching) == 1)
            if alternative is not None:
                assert not np.array_equal(alternative, TARGET)
                assert any(np.array_equal(alternative, grid) for grid in matching)


def test_known_four_clue_puzzle_and_timeout():
    puzzle = np.array([[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 4, 0], [2, 0, 0, 0]])
    assert find_alternative(TARGET, puzzle)[0] == "infeasible"
    assert find_alternative(TARGET, puzzle, time_limit=0)[0] == "limit"
    with pytest.raises(ValueError, match="disagree"):
        find_alternative(TARGET, np.full((4, 4), 1))
    with pytest.raises(ValueError, match="valid completed"):
        find_alternative(np.zeros((4, 4), dtype=int), np.zeros((4, 4), dtype=int))


def test_generated_twenty_one_clue_example():
    path = Path(__file__).resolve().parents[1] / "examples/smoke_9x9.json"
    saved = json.loads(path.read_text())
    target = np.array(saved["target"]) - 1
    puzzle = np.array(saved["puzzle"])
    assert np.count_nonzero(puzzle) == saved["clue_count"] == 21
    assert find_alternative(target, puzzle, time_limit=2)[0] == "infeasible"
