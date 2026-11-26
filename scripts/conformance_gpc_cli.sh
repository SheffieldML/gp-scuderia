#!/bin/sh
# CIP-0016 A2: the gpc CLI imports and exports interchange documents faithfully.
# Usage: conformance_gpc_cli.sh <GPc root containing built ./gp> <examples dir> [python]
set -eu
GPC=$1; EX=$2; PY=${3:-python3}
GP="$GPC/gp"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$T"
fail=0
ok() { echo "ok   $1"; }
bad() { echo "FAIL $1"; fail=1; }

# import then export reproduces the sparse golden's structure and parameters
"$GP" -v 0 import "$EX/dtcvar_ard_data.json" m.model
"$GP" -v 0 export m.model.csv m.model out.json
"$PY" - "$EX/dtcvar_ard_data.json" out.json <<'P' && ok "import/export round trip (dtcvar)" || bad "import/export round trip (dtcvar)"
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
sys.exit(0 if all(a[k] == b[k] for k in ("kernel", "likelihood", "inducing_inputs", "data", "approximation", "sparse_linalg")) else 1)
P

# exact GP with White cannot round trip through .model: import must refuse
if "$GP" -v 0 import "$EX/ftc_rbf_white.json" w.model >/dev/null 2>&1 </dev/null; then bad "import refuses exact+White"; else ok "import refuses exact+White"; fi

# learning with default centring: export carries the bias as normalizer.center
printf '0,0.1\n1,0.9\n2,0.2\n3,-0.5\n' > d.csv
"$GP" -v 0 learn d.csv c.model >/dev/null 2>&1 </dev/null || true
"$GP" -v 0 export d.csv c.model c.json >/dev/null 2>&1 </dev/null && ok "export accepts centred targets" || bad "export accepts centred targets"
"$PY" - c.json <<'P' && ok "centred export carries normalizer.center = mean(y)" || bad "centred export carries normalizer.center = mean(y)"
import json, sys
n = json.load(open(sys.argv[1]))["normalizer"]
sys.exit(0 if n["type"] == "normalizer.center" and abs(n["mean"] - 0.175) < 1e-9 else 1)
P
"$GP" -v 0 learn -C 0 -S 0 d.csv u.model >/dev/null 2>&1 </dev/null
"$GP" -v 0 export d.csv u.model u.json && ok "export accepts uncentred targets" || bad "export accepts uncentred targets"
exit $fail
