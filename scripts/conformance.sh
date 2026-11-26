#!/bin/sh
# Run every toolbox adapter against the goldens' GPy-generated x-reference.
# Environment overrides: GPC_PY (GPc/python), GPMAT, ANACONDA_PY, HDF5_LIB, GPY_PY.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPC_PY=${GPC_PY:-$HOME/SheffieldML/GPc/python}
PY=${ANACONDA_PY:-/opt/anaconda3/bin/python}
status=0
echo "== GPc C++ (CInterchange) =="
(cd "${GPC_ROOT:-$GPC_PY/..}" && make -s testInterchange >/dev/null && ./testInterchange "$ROOT/examples") || status=1
echo "== GPc CLI (gp import/export) =="
(cd "${GPC_ROOT:-$GPC_PY/..}" && make -s gp >/dev/null) && sh "$ROOT/scripts/conformance_gpc_cli.sh" "${GPC_ROOT:-$GPC_PY/..}" "$ROOT/examples" "$PY" || status=1
echo "== GPc =="
PYTHONPATH="$GPC_PY:$ROOT/python" "$PY" "$ROOT/scripts/conformance_gpc.py" || status=1
echo "== GPy (to_wire / from_wire) =="
"${GPY_PY:-$ROOT/.venv-gpy/bin/python}" "$ROOT/scripts/conformance_gpy.py" 2>/dev/null || status=1
echo "== Laplace goldens vs independent numpy implementation =="
"${GPY_PY:-$ROOT/.venv-gpy/bin/python}" "$ROOT/scripts/check_laplace_independent.py" 2>/dev/null || status=1
for fw in gpflow gpytorch gpjax; do
  echo "== $fw adapter =="
  if [ -x "$ROOT/.venv-$fw/bin/python" ]; then
    "$ROOT/.venv-$fw/bin/python" "$ROOT/scripts/check_adapter.py" $fw 2>/dev/null || status=1
  else echo "skipped (no .venv-$fw)"; fi
done
echo "== GPmat (Octave) =="
DYLD_FALLBACK_LIBRARY_PATH="${HDF5_LIB:-/opt/homebrew/Cellar/hdf5/1.14.5/lib:/opt/homebrew/lib}" \
  octave-cli --quiet "$ROOT/octave/conformance_gpmat.m" || status=1
exit $status
