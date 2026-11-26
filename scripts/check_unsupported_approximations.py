"""Every adapter must refuse approximation tags that have no golden yet (dtc, pitc).

Usage: .venv-<fw>/bin/python scripts/check_unsupported_approximations.py gpy|gpflow|gpytorch|gpjax
The schema allows these tags and the registry records them; a missing adapter
path must surface as UnsupportedByFramework, never as a wrong number or a
generic exception.
"""
import copy, importlib, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load
from gp_scuderia._common import UnsupportedByFramework

A = importlib.import_module("gp_scuderia.%s_adapter" % sys.argv[1])
base = load(root / "examples" / "fitc_matern52.yaml")
bad = 0
for tag in ("dtc", "pitc"):
    doc = copy.deepcopy(base); doc["approximation"] = tag
    try:
        A.reference(doc, doc["x-reference"]["Xnew"])
        print("FAIL %s: %s ran without refusing" % (sys.argv[1], tag)); bad += 1
    except UnsupportedByFramework:
        print("ok   %s refuses approximation %s" % (sys.argv[1], tag))
    except Exception as e:
        print("FAIL %s: %s raised %s, not UnsupportedByFramework" % (sys.argv[1], tag, type(e).__name__)); bad += 1
sys.exit(bad > 0)
