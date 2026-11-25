"""Generate x-reference blocks in the goldens using GPy as the oracle.

Run with .venv-gpy/bin/python. Rewrites YAML and JSON twins.
"""
import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load, dump, to_json_twin
from gp_scuderia import gpy_adapter

XNEW = {1: [[-1.5], [0.5], [3.0]], 2: [[-1.0, 0.0], [0.5, 0.5], [2.0, 1.5]]}

for name in ["ftc_rbf_white", "dtcvar_ard_data"]:
    p = root / "examples" / (name + ".yaml")
    doc = load(p)
    doc.pop("x-reference", None)
    ref = gpy_adapter.reference(doc, XNEW[doc["input_dim"]])
    ref["oracle"] = "GPy"
    # GPmat adds 1e-6 jitter to K_uu on sparse models; GPy and GPc add none.
    # Measured effect on the DTCVar bound is ~4e-5, hence the log-likelihood tolerance.
    ref["tolerance"] = {"log_likelihood": 1e-4, "mean": 1e-7, "var": 1e-5}
    doc["x-reference"] = ref
    dump(doc, p)
    to_json_twin(p)
    print(name, ref["log_likelihood"], ref["mean"])
