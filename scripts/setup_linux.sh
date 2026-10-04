#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import gurobipy as gp; e = gp.Env(empty=True); e.setParam("OutputFlag", 0); e.start(); print("Gurobi", gp.gurobi.version(), "license ready"); e.dispose()'
