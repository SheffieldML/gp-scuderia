"""Check one framework adapter against every golden's x-reference.

Usage: .venv-<fw>/bin/python scripts/check_adapter.py gpflow|gpytorch|gpjax
Goldens a framework cannot realise must raise UnsupportedByFramework; that is
reported as 'refuses', not as a failure.
"""
import importlib, pathlib, sys
import numpy as np
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load
from gp_scuderia._common import UnsupportedByFramework

fw = sys.argv[1]
A = importlib.import_module("gp_scuderia.%s_adapter" % fw)
EXPECT_REFUSE = {  # goldens the framework is expected to refuse, with the reason
    "gpflow": {"ftc_icm", "ftc_lcm", "fitc_matern52"},
    "gpytorch": {"ftc_rbf_white", "ftc_icm", "ftc_lcm", "fitc_matern52", "dtcvar_ard_data", "dtcvar_mean_constant"},
    "gpjax": {"ftc_icm", "ftc_lcm", "fitc_matern52"},
}[fw]
bad = 0
for p in sorted((root / "examples").glob("*.yaml")):
    name = p.stem
    doc = load(p); ref = doc["x-reference"]; tol = ref["tolerance"]
    try:
        g = A.reference(doc, ref["Xnew"])
    except UnsupportedByFramework as e:
        ok = name in EXPECT_REFUSE
        print(("ok   " if ok else "FAIL ") + "%s refuses (%s)" % (name, e)); bad += (not ok); continue
    if name in EXPECT_REFUSE:
        print("FAIL %s expected to be refused but ran" % name); bad += 1; continue
    dl = abs(g["log_likelihood"] - ref["log_likelihood"])
    dm = np.abs(np.array(g["mean"]) - ref["mean"]).max()
    dv = np.abs(np.array(g["var"]) - ref["var"]).max()
    ok = dl <= tol["log_likelihood"] and dm <= tol["mean"] and dv <= tol["var"]
    print(("ok   " if ok else "FAIL ") + "%s matches GPy: dLL=%.2g dmean=%.2g dvar=%.2g" % (name, dl, dm, dv))
    bad += (not ok)
sys.exit(bad > 0)
