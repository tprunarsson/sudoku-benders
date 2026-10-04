#!/usr/bin/env python3
"""Compute representatives of the essentially different top bands of an N x N Sudoku.

A top band is the first sqrt(N) rows of a grid. Two bands are equivalent if one
can be turned into the other by permuting stacks, permuting columns inside
stacks, permuting the band's rows and relabelling digits. Every grid is
equivalent to a grid whose top band is one of the representatives, because band
permutations can move any of its bands to the top and all of the operations
above are Sudoku symmetries.

Bands are normalised so that box 1 holds 1..N in reading order. The
representative of a class is its lexicographically smallest normalised band.
For N=9 there are 416 classes, for N=4 there are 2.

Output: top_bands_<N>.json, a list of [band string of b*N digits, class size].
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def enumerate_bands(n: int):
    b = math.isqrt(n)
    box1 = [list(range(r * b + 1, r * b + b + 1)) for r in range(b)]
    rows_rest = [sorted(set(range(1, n + 1)) - set(row)) for row in box1]

    def rec(r: int, chosen: list[tuple[int, ...]]):
        if r == b:
            yield bytes(d for row in range(b) for d in box1[row] + list(chosen[row]))
            return
        for perm in itertools.permutations(rows_rest[r]):
            ok = True
            for s in range(1, b):
                seg = set(perm[(s - 1) * b:s * b])
                if any(seg & set(prev[(s - 1) * b:s * b]) for prev in chosen):
                    ok = False
                    break
            if ok:
                yield from rec(r + 1, chosen + [perm])

    yield from rec(0, [])


def generators(n: int) -> list[tuple[int, ...]]:
    b = math.isqrt(n)

    def index(rowp, colp):
        return tuple(rowp[r] * n + colp[c] for r in range(b) for c in range(n))

    rows = list(range(b))
    cols = list(range(n))
    swap01 = {0: 1, 1: 0}
    stack_swap = [swap01.get(c // b, c // b) * b + c % b for c in cols]
    stack_cycle = [(((c // b) + 1) % b) * b + c % b for c in cols]
    col_swap = [1, 0] + cols[2:]
    col_cycle = [(c + 1) % b if c < b else c for c in cols]
    row_swap = [1, 0] + rows[2:]
    row_cycle = [(r + 1) % b for r in rows]
    return [
        index(rows, stack_swap),
        index(rows, stack_cycle),
        index(rows, col_swap),
        index(rows, col_cycle),
        index(row_swap, cols),
        index(row_cycle, cols),
    ]


def classes(n: int) -> list[tuple[bytes, int]]:
    b = math.isqrt(n)
    box1 = [r * n + c for r in range(b) for c in range(b)]
    target = bytes(range(1, n + 1))
    gens = generators(n)

    def normalise(band: bytes) -> bytes:
        return band.translate(bytes.maketrans(bytes(band[i] for i in box1), target))

    seen: set[bytes] = set()
    out = []
    for start in enumerate_bands(n):
        if start in seen:
            continue
        orbit, stack = {start}, [start]
        while stack:
            band = stack.pop()
            for g in gens:
                nb = normalise(bytes(band[i] for i in g))
                if nb not in orbit:
                    orbit.add(nb)
                    stack.append(nb)
        seen |= orbit
        out.append((min(orbit), len(orbit)))
    out.sort()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", "-n", type=int, default=9, choices=[4, 9])
    args = p.parse_args()
    result = classes(args.size)
    path = Path(__file__).with_name(f"top_bands_{args.size}.json")
    path.write_text(json.dumps([["".join(map(str, band)), size] for band, size in result]))
    print(f"{len(result)} classes covering {sum(s for _, s in result)} bands -> {path}")


if __name__ == "__main__":
    main()
