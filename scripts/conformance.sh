#!/bin/sh
# Run every toolbox adapter against the goldens' GPy-generated x-reference.
# Tool locations come from the environment; sections whose tool is not set are skipped.
#   GPC_ROOT   GPc checkout (built in place with make); enables the C++, CLI and Python GPc checks
#   GPC_PY     GPc Python package directory (default: $GPC_ROOT/python)
#   PYTHON     interpreter with numpy, pyyaml (and the gpc package for GPc), default python3
#   GPY_PY     interpreter with GPy installed (default: ./.venv-gpy/bin/python, else $PYTHON)
#   GPMAT      GPmat checkout; enables the Octave check (needs octave-cli)
#   GPML       GPML checkout (python scripts/ci_fetch.py gpml <dir> fetches it); enables the GPML comparison
#   HDF5_LIB   extra library path for Octave on macOS, optional
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
GPC_ROOT=${GPC_ROOT:-}
GPC_PY=${GPC_PY:-${GPC_ROOT:+$GPC_ROOT/python}}
if [ -x "$ROOT/.venv-gpy/bin/python" ]; then GPY_DEFAULT="$ROOT/.venv-gpy/bin/python"; else GPY_DEFAULT=$PY; fi
GPY_PY=${GPY_PY:-$GPY_DEFAULT}
status=0
if [ -n "$GPC_ROOT" ]; then
echo "== GPc C++ (CInterchange) =="
(cd "$GPC_ROOT" && make -s testInterchange >/dev/null && ./testInterchange "$ROOT/examples") || status=1
echo "== GPc CLI (gp import/export) =="
(cd "$GPC_ROOT" && make -s gp >/dev/null) && sh "$ROOT/scripts/conformance_gpc_cli.sh" "$GPC_ROOT" "$ROOT/examples" "$PY" || status=1
echo "== GPc (Python) =="
if [ -d "$GPC_PY/gpc" ]; then
  PYTHONPATH="$GPC_PY:$ROOT/python" "$PY" "$ROOT/scripts/conformance_gpc.py" || status=1
else echo "skipped (no gpc package under $GPC_PY; build it with make -C python)"; fi
else echo "== GPc ==" ; echo "skipped (GPC_ROOT not set)"; fi
echo "== GPy (to_wire / from_wire) =="
"$GPY_PY" "$ROOT/scripts/conformance_gpy.py" 2>/dev/null || status=1
echo "== Framework files (frameworks/*.yaml) =="
"$GPY_PY" "$ROOT/scripts/check_framework_files.py" || status=1
echo "== Runner contract (stub framework) =="
"$GPY_PY" "$ROOT/scripts/check_runner_contract.py" || status=1
echo "== Laplace goldens vs independent numpy implementation =="
"$GPY_PY" "$ROOT/scripts/check_laplace_independent.py" 2>/dev/null || status=1
"$GPY_PY" "$ROOT/scripts/check_unsupported_approximations.py" gpy 2>/dev/null || status=1
BENCH=$(mktemp -d)
"$GPY_PY" "$ROOT/scripts/benchmarks.py" generate --out "$BENCH" >/dev/null || status=1
# Every framework with a pinned entry in frameworks/*.yaml (the same list CI uses).
"$GPY_PY" "$ROOT/scripts/ci_matrix.py" --lines | while read fw kind env; do
  echo "== $fw =="
  case $kind in
  python)
    if [ -x "$ROOT/.venv-$fw/bin/python" ]; then
      "$ROOT/.venv-$fw/bin/python" "$ROOT/scripts/compare.py" $fw || echo "FAILED $fw" >> "$ROOT/.conformance-failed"
      "$ROOT/.venv-$fw/bin/python" "$ROOT/scripts/benchmarks.py" run $fw --data "$BENCH" || echo "FAILED $fw benchmarks" >> "$ROOT/.conformance-failed"
      "$ROOT/.venv-$fw/bin/python" "$ROOT/scripts/check_unsupported_approximations.py" $fw 2>/dev/null || echo "FAILED $fw unsupported" >> "$ROOT/.conformance-failed"
    else echo "skipped (no .venv-$fw)"; fi ;;
  octave)
    eval "dir=\${$env:-}"
    if [ -n "$dir" ] && command -v octave-cli >/dev/null 2>&1; then
      if [ "$fw" = gpmat ]; then
        DYLD_FALLBACK_LIBRARY_PATH="${HDF5_LIB:-}" octave-cli --quiet "$ROOT/octave/conformance_gpmat.m" || echo "FAILED gpmat conformance" >> "$ROOT/.conformance-failed"
      fi
      DYLD_FALLBACK_LIBRARY_PATH="${HDF5_LIB:-}" "$PY" "$ROOT/scripts/compare.py" $fw || echo "FAILED $fw" >> "$ROOT/.conformance-failed"
      DYLD_FALLBACK_LIBRARY_PATH="${HDF5_LIB:-}" "$PY" "$ROOT/scripts/benchmarks.py" run $fw --data "$BENCH" || echo "FAILED $fw benchmarks" >> "$ROOT/.conformance-failed"
    else echo "skipped (set $env and install octave-cli)"; fi ;;
  esac
done
if [ -f "$ROOT/.conformance-failed" ]; then cat "$ROOT/.conformance-failed"; rm -f "$ROOT/.conformance-failed"; status=1; fi
exit $status
