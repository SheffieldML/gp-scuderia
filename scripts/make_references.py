"""Generate x-reference blocks in the goldens using GPy as the oracle.

Run with .venv-gpy/bin/python. Rewrites YAML and JSON twins.
"""
import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load, dump, to_json_twin
from gp_scuderia import gpy_adapter

XNEW = {1: [[-1.5], [0.5], [3.0]], 2: [[-1.0, 0.0], [0.5, 0.5], [2.0, 1.5]]}

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
             "ftc_standardize", "dtcvar_mean_constant"]:
    p = root / "examples" / (name + ".yaml")
    doc = load(p)
    doc.pop("x-reference", None)
    ref = gpy_adapter.reference(doc, XNEW[doc["input_dim"]])
    ref["oracle"] = "GPy"
    # GPmat adds 1e-6 jitter to K_uu on sparse models; GPy and GPc add none.
    # Measured effect on the DTCVar bound is ~4e-5, hence the log-likelihood tolerance.
    ref["tolerance"] = {"log_likelihood": 1e-4, "mean": 1e-7, "var": 1e-5}
    if doc.get("approximation") == "fitc":
        # GPmat agrees with GPy to 1e-15 here; GPc differs by ~3e-7 on the mean
        # (jittered Cholesky of K_uu in the whitened FITC path), so this golden
        # states a wider mean tolerance.
        ref["tolerance"]["mean"] = 1e-6
    doc["x-reference"] = ref
    dump(doc, p)
    to_json_twin(p)
    print(name, ref["log_likelihood"], ref["mean"])
