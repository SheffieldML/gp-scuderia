#!/usr/bin/env python3
"""Generate the mean.linear_basis goldens (CIP-0009) with the NumPy reference as oracle.

GPy has no native estimated-beta modes, so x-reference comes from
gp_scuderia.linear_basis, which scripts/check_linear_basis.py checks against
independent computations. Needs numpy and pyyaml only.
"""
import pathlib, sys
import numpy as np
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import dump, to_json_twin, linear_basis as lb

rng = np.random.default_rng(20261128)
r6 = lambda a: np.round(np.asarray(a, float), 6).tolist()
BASE = {"schema_version": "0.1", "type": "models.gp_regression", "output_dim": 1, "approximation": "ftc",
        "likelihood": {"type": "likelihoods.gaussian", "variance": 0.09}}
LIN2 = [[0, 0], [1, 0], [0, 1]]
QUAD1 = [[0], [1], [2]]


def data(n, d, trend, noise=0.3):
    X = rng.uniform(-2, 2, size=(n, d))
    F = lb.polynomial_basis(X, trend["powers"])
    Y = F @ np.asarray(trend["beta"]) + noise * rng.normal(size=n)
    return r6(X), r6(Y)


GOLDENS = {}
X2, Y2 = data(8, 2, {"powers": LIN2, "beta": [1.5, -0.8, 0.6]})
X1, Y1 = data(7, 1, {"powers": QUAD1, "beta": [0.5, 1.0, -0.4]})
k2 = {"type": "kern.rbf", "variance": 1.1, "lengthscale": 1.4, "ARD": False}
k1 = {"type": "kern.rbf", "variance": 0.8, "lengthscale": 1.2, "ARD": False}
for name, beta in [("ftc_linear_fixed", {"mode": "fixed", "value": [1.2, -0.5, 0.4]}),
                   ("ftc_linear_profiled", {"mode": "profiled"}),
                   ("ftc_linear_flat_prior", {"mode": "flat_prior"}),
                   ("ftc_linear_gaussian_prior", {"mode": "gaussian_prior", "prior_mean": [0.5, 0.0, 0.0],
                                                  "prior_cov": [[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]]})]:
    GOLDENS[name] = dict(BASE, input_dim=2, kernel=k2, data={"X": X2, "Y": [[y] for y in Y2]},
                         mean={"type": "mean.linear_basis", "basis": {"type": "basis.polynomial", "powers": LIN2},
                               "beta": beta})
GOLDENS["ftc_linear_quadratic_profiled"] = dict(
    BASE, input_dim=1, kernel=k1, data={"X": X1, "Y": [[y] for y in Y1]},
    mean={"type": "mean.linear_basis", "basis": {"type": "basis.polynomial", "powers": QUAD1},
          "beta": {"mode": "profiled"}})
XNEW = {2: [[-1.0, 0.0], [0.5, 0.5], [2.0, 1.5]], 1: [[-1.5], [0.5], [3.0]]}

for name, doc in GOLDENS.items():
    # data rows are lists of floats; X is already [[..]] for d = 2 and needs nesting for d = 1
    if doc["input_dim"] == 1:
        doc["data"]["X"] = [[x] if not isinstance(x, list) else x for x in doc["data"]["X"]]
    ref = lb.reference_from_doc(doc, XNEW[doc["input_dim"]])
    ref.pop("beta")
    ref["oracle"] = "numpy reference gp_scuderia.linear_basis (CIP-0009)"
    ref["tolerance"] = {"log_likelihood": 1e-7, "mean": 1e-7, "var": 1e-7}
    doc["x-reference"] = ref
    p = root / "examples" / (name + ".yaml")
    dump(doc, p)
    to_json_twin(p)
    print(name, ref["log_likelihood"])
