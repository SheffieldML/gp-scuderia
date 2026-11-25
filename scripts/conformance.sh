#!/bin/sh
# Run every toolbox adapter against the goldens' GPy-generated x-reference.
# Environment overrides: GPC_PY (GPc/python), GPMAT, ANACONDA_PY, HDF5_LIB.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GPC_PY=${GPC_PY:-$HOME/SheffieldML/GPc/python}
PY=${ANACONDA_PY:-/opt/anaconda3/bin/python}
status=0
echo "== GPc C++ (CInterchange) =="
(cd "${GPC_ROOT:-$GPC_PY/..}" && make -s testInterchange >/dev/null && ./testInterchange "$ROOT/examples") || status=1
echo "== GPc =="
PYTHONPATH="$GPC_PY:$ROOT/python" "$PY" "$ROOT/scripts/conformance_gpc.py" || status=1
echo "== GPmat (Octave) =="
DYLD_FALLBACK_LIBRARY_PATH="${HDF5_LIB:-/opt/homebrew/Cellar/hdf5/1.14.5/lib:/opt/homebrew/lib}" \
  octave-cli --quiet "$ROOT/octave/conformance_gpmat.m" || status=1
exit $status
