import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import sudoku_benders_gurobi as benders
from sudoku_search import find_alternative
from sudoku_run_support import RunRecorder, greedy_start


@pytest.mark.parametrize("mode,cut,warm", [
    ("callback", "unavoidable", True),
    ("callback", "delta", False),
    ("iterative", "both", True),
])
def test_small_model_proves_optimum_and_writes_verified_best(tmp_path, mode, cut, warm):
    result = benders.run_benders(n=4, mode=mode, cut=cut, separator="hybrid",
                                row_symmetry=True, digit_cover=True, line_pairs=True,
                                rectangles="lazy", warm_start=warm, run_dir=tmp_path,
                                threads=2, subproblem_threads=1, time_limit=20, verbose=False,
                                max_master_solves=1000)
    assert result.status == "optimal"
    assert result.clue_count == result.lower_bound == 4
    assert result.verified_unique is True
    assert find_alternative(result.target - 1, result.puzzle)[0] == "infeasible"
    saved = json.loads((tmp_path / "best.json").read_text())
    assert saved["clue_count"] == 4
    assert saved["verified_unique"] is True
    assert (tmp_path / "progress.json").is_file()


def test_callback_errors_are_propagated(monkeypatch):
    def failing_separator(*args, **kwargs):
        raise RuntimeError("separation failed")
    monkeypatch.setattr(benders, "separate", failing_separator)
    with pytest.raises(RuntimeError, match="separation failed"):
        benders.run_benders(n=4, mode="callback", threads=1, verbose=False, time_limit=2)


def test_expired_deadline_never_calls_separator():
    import time
    class NoSolve:
        def solve(self, *args, **kwargs):
            raise AssertionError("budget already exhausted")
    target = np.zeros((4, 4), dtype=int)
    clues = np.zeros((4, 4, 4), dtype=int)
    first, batch, limited = benders.separate(NoSolve(), target, clues, 4, "hybrid", None,
                                             time.perf_counter() - 1)
    assert first is None and batch == [] and limited


def test_saved_start_is_rechecked_and_reused(tmp_path):
    first = benders.run_benders(n=4, warm_start=True, row_symmetry=True,
                               run_dir=tmp_path / "first", threads=1,
                               verbose=False, time_limit=10)
    second = benders.run_benders(n=4, row_symmetry=True, start_json=tmp_path / "first/best.json",
                                run_dir=tmp_path / "second", threads=1,
                                verbose=False, time_limit=10)
    assert second.clue_count == first.clue_count == 4
    assert second.verified_unique


def test_greedy_start_keeps_uniqueness_and_known_bound(tmp_path):
    import time
    target = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]])
    recorder = RunRecorder(tmp_path, time.perf_counter(), None, verbose=False)
    clues, witnesses = greedy_start(target, recorder, seed=7, budget=2, min_clues=5, spread={})
    assert int(clues.sum()) == 5
    assert recorder.best[2] == 5
    assert find_alternative(target, benders.clue_grid(4, clues))[0] == "infeasible"
    for alternative in witnesses:
        assert not np.array_equal(target, alternative)


def test_assumed_bound_and_infeasible_requirements_are_respected(tmp_path):
    result = benders.run_benders(n=4, min_clues=3, warm_start=True, threads=1,
                                run_dir=tmp_path / "bounded", time_limit=10, verbose=False)
    assert result.lower_bound == 4
    assert all(event["master_bound"] >= 3 for event in result.history)
    impossible = benders.run_benders(n=4, min_clues=17, warm_start=True, threads=1,
                                     run_dir=tmp_path / "impossible", time_limit=10, verbose=False)
    assert impossible.status == "infeasible"
    assert impossible.clue_count is None
    assert not (tmp_path / "impossible/best.json").exists()


def test_timeout_returns_saved_best_instead_of_last_candidate(tmp_path, monkeypatch):
    import time
    original = benders.run_callback
    def expired_callback(master, sub, cut, batch_size, separator, sub_limit, verbose,
                         start, deadline, recorder):
        # Exercise the normal optimizer once, then emulate the budget expiring
        # before the separate final verification can start.
        state = original(master, sub, cut, batch_size, separator, sub_limit, verbose,
                         start, deadline, recorder)
        recorder.deadline = time.perf_counter() - 1
        target, clues, _ = master.values()
        # A last returned master candidate can be unverified when termination
        # happens inside MIPSOL. It must not replace the durable incumbent.
        master.values = lambda source=None: (target, np.zeros_like(clues), 0)
        return state
    monkeypatch.setattr(benders, "run_callback", expired_callback)
    result = benders.run_benders(n=4, warm_start=True, threads=1, time_limit=10,
                                run_dir=tmp_path, verbose=False)
    assert result.clue_count == 4 and result.verified_unique is True
    assert result.status == "optimal"
    assert json.loads((tmp_path / "best.json").read_text())["clue_count"] == 4
