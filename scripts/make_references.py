"""Generate x-reference blocks in the goldens using GPy as the oracle.

Run with .venv-gpy/bin/python. Rewrites YAML and JSON twins.
"""
import pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load, dump, to_json_twin
from gp_scuderia import gpy_adapter

XNEW = {1: [[-1.5], [0.5], [3.0]], 2: [[-1.0, 0.0], [0.5, 0.5], [2.0, 1.5]]}

# Multi-output goldens: the last column of Xnew is the output index.
XNEW_BY_NAME = {"ftc_icm": [[0.7, 0], [0.7, 1], [3.0, 1]],
                "ftc_lcm": [[0.7, 0], [0.7, 1], [0.7, 2], [3.0, 2]]}

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
             "ftc_standardize", "dtcvar_mean_constant", "ftc_icm", "ftc_lcm",
             "laplace_bernoulli", "laplace_poisson"]:
    p = root / "examples" / (name + ".yaml")
    doc = load(p)
    doc.pop("x-reference", None)
    ref = gpy_adapter.reference(doc, XNEW_BY_NAME.get(name, XNEW[doc["input_dim"]]))
    ref["oracle"] = "GPy"
    # GPmat adds 1e-6 jitter to K_uu on sparse models; GPy and GPc add none.
    # Measured effect on the DTCVar bound is ~4e-5, hence the log-likelihood tolerance.
    ref["tolerance"] = {"log_likelihood": 1e-4, "mean": 1e-7, "var": 1e-5}
    if doc.get("approximation") == "fitc":
        # GPmat agrees with GPy to 1e-15 here; GPc differs by ~3e-7 on the mean
        # (jittered Cholesky of K_uu in the whitened FITC path), so this golden
        # states a wider mean tolerance.
        ref["tolerance"]["mean"] = 1e-6
    if doc.get("inference") == "laplace":
        # Checked against an independent numpy Laplace (scripts/check_laplace_independent.py).
        ref["tolerance"] = {"log_likelihood": 1e-6, "mean": 1e-6, "var": 1e-6}
        ref["oracle"] = "GPy (cross-checked: numpy Laplace, Rasmussen and Williams Alg 3.1/3.2)"
    doc["x-reference"] = ref
    dump(doc, p)
    to_json_twin(p)
    print(name, ref["log_likelihood"], ref["mean"])


# Latent-variable goldens (CIP-0006) live in examples/latent/: their reference is the bound and
# the psi statistics, not a prediction, so the prediction-based tools do not iterate over them.
import numpy as np


def make_bgplvm_rbf_ard():
    """Fixed, hand-readable inputs (seeded, rounded to 3 places) for the collapsed Bayesian GP-LVM."""
    rng = np.random.default_rng(6)
    n, q, d, m = 8, 2, 3, 4
    mu = np.round(rng.standard_normal((n, q)), 3)
    S = np.round(rng.uniform(0.1, 0.6, (n, q)), 3)
    Z = np.round(rng.standard_normal((m, q)), 3)
    W = rng.standard_normal((q, d))
    Y = np.round(np.sin(mu @ W) + 0.1 * rng.standard_normal((n, d)), 3)
    return {"schema_version": "0.1", "type": "models.bayesian_gplvm", "approximation": "dtcvar",
            "inducing_inputs": Z.tolist(),
            "kernel": {"type": "kern.rbf", "variance": 1.5, "lengthscale": [0.9, 1.7], "ARD": True},
            "likelihood": {"type": "likelihoods.gaussian", "variance": 0.05},
            "latent": {"dim": q, "mean": mu.tolist(), "variance": S.tolist(), "prior": {"type": "standard_normal"}},
            "data": {"Y": Y.tolist()}}


p = root / "examples" / "latent" / "bgplvm_rbf_ard.yaml"
p.parent.mkdir(exist_ok=True)
doc = make_bgplvm_rbf_ard()
b, ps = gpy_adapter.bound(doc), gpy_adapter.psi(doc)
doc["x-reference"] = {**b, "psi0": np.asarray(ps["psi0"]).tolist(), "psi1": np.asarray(ps["psi1"]).tolist(),
                      "psi2": np.asarray(ps["psi2"]).tolist(), "oracle": "GPy",
                      # GPmat adds 1e-6 jitter to K_uu (see above): ~4e-5 on the bound; kl and psi do not use K_uu
                      "tolerance": {"bound_without_kl": 1e-4, "kl": 1e-10, "psi": 1e-8}}
dump(doc, p)
to_json_twin(p)
print("bgplvm_rbf_ard", b)
