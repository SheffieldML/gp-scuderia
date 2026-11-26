"""GPy adapter conformance: the goldens round trip through GPy and still match x-reference.

Run: .venv-gpy/bin/python scripts/conformance_gpy.py
"""
import copy, json, pathlib, sys
from jsonschema import Draft202012Validator
import numpy as np
import GPy
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load, gpy_adapter as A

bad = 0
validator = Draft202012Validator(json.loads((root / 'schemas/model.schema.json').read_text()))
def check(cond, msg):
    global bad
    print(("ok   " if cond else "FAIL ") + msg)
    bad += (not cond)

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
             "ftc_standardize", "dtcvar_mean_constant", "ftc_icm", "ftc_lcm"]:
    doc = load(root / "examples" / (name + ".yaml"))
    ref = doc["x-reference"]; tol = ref["tolerance"]
    m = A.from_wire(doc)
    r = A.reference(doc, ref["Xnew"])
    check(abs(r["log_likelihood"] - ref["log_likelihood"]) <= 1e-9
          and np.abs(np.array(r["mean"]) - ref["mean"]).max() <= 1e-9,
          "%s reproduces its own x-reference" % name)
    back = A.to_wire(m)
    same = all(back[k] == doc[k] for k in
               ("schema_version", "type", "input_dim", "output_dim", "approximation",
                "kernel", "likelihood", "data") + (("inducing_inputs",) if "inducing_inputs" in doc else ()))
    check(same, "%s round trips through to_wire exactly" % name)
    errs = list(validator.iter_errors(back))
    check(not errs, "%s to_wire output validates against the schema" % name)
    m2 = A.from_wire(back)
    check(abs(m2.log_likelihood() - m.log_likelihood()) < 1e-12, "%s rebuilt model has the same log likelihood" % name)

X = np.random.RandomState(0).randn(6, 1); Y = np.sin(X)
def refuses(model, label):
    try:
        A.to_wire(model); check(False, "refuses " + label)
    except A.WireExportError as e:
        check(True, "refuses %s (%s)" % (label, e))
refuses(GPy.models.GPRegression(X, Y, GPy.kern.StdPeriodic(1)), "unsupported kernel")
refuses(GPy.models.GPRegression(X, Y, GPy.kern.RBF(1), normalizer=True,
                                mean_function=GPy.mappings.Constant(1, 1)), "mean with normalizer")
refuses(GPy.models.GPRegression(X, Y, GPy.kern.RBF(1), mean_function=GPy.mappings.Linear(1, 1)), "non-constant mean function")
refuses(GPy.models.GPRegression(X, np.hstack([Y, Y]), GPy.kern.RBF(1)), "multi-output")
refuses(GPy.models.GPClassification(X, (Y > 0).astype(float), GPy.kern.RBF(1)), "non-Gaussian model")
_mx = GPy.core.GP(np.hstack([X, X*0]), Y, GPy.kern.RBF(1, active_dims=[0]),
                  GPy.likelihoods.MixedNoise([GPy.likelihoods.Gaussian(variance=0.1)]),
                  Y_metadata={"output_index": np.zeros((6, 1), int)})
refuses(_mx, "mixed noise without a coregionalize index column")
# Further shapes: FITC, matern52 and a product kernel survive the round trip.
doc = {"schema_version": "0.1", "type": "models.sparse_gp_regression", "input_dim": 1,
       "output_dim": 1, "approximation": "fitc", "inducing_inputs": [[-1.0], [0.0], [1.0]],
       "kernel": {"type": "kern.prod", "parts": [
           {"type": "kern.matern52", "variance": 1.3, "lengthscale": 0.7, "ARD": False},
           {"type": "kern.rbf", "variance": 0.9, "lengthscale": 2.0, "ARD": False}]},
       "likelihood": {"type": "likelihoods.gaussian", "variance": 0.1},
       "data": {"X": X.tolist(), "Y": Y.tolist()}}
back = A.to_wire(A.from_wire(doc))
check(all(back[k] == doc[k] for k in doc) and not list(validator.iter_errors(back)),
      "fitc + prod(matern52, rbf) round trips and validates")
sys.exit(1 if bad else 0)
