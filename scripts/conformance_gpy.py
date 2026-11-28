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
from gp_scuderia._common import UnsupportedByFramework

bad = 0
validator = Draft202012Validator(json.loads((root / 'schemas/model.schema.json').read_text()))
def check(cond, msg):
    global bad
    print(("ok   " if cond else "FAIL ") + msg)
    bad += (not cond)

def raises(f, exc, label):
    try:
        f(); check(False, "raises " + label)
    except exc as e:
        check(True, "raises %s (%s)" % (label, e))

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
             "ftc_standardize", "dtcvar_mean_constant", "ftc_icm", "ftc_lcm",
             "laplace_bernoulli", "laplace_poisson", "ftc_linear_fixed", "ftc_linear_intercept_fixed"]:
    doc = load(root / "examples" / (name + ".yaml"))
    ref = doc["x-reference"]; tol = ref["tolerance"]
    m = A.from_wire(doc)
    r = A.reference(doc, ref["Xnew"])
    # Laplace mode-finding is iterative, so BLAS differences move results by ~1e-8; use the golden's
    # own tolerance there and stay tight for the closed-form cases.
    lt, mt = (tol["log_likelihood"], tol["mean"]) if name.startswith(("laplace", "ftc_linear")) else (1e-9, 1e-9)
    check(abs(r["log_likelihood"] - ref["log_likelihood"]) <= lt
          and np.abs(np.array(r["mean"]) - ref["mean"]).max() <= mt,
          "%s reproduces its own x-reference" % name)
    back = A.to_wire(m)
    same = all(back[k] == doc[k] for k in
               ("schema_version", "type", "input_dim", "output_dim", "approximation",
                "kernel", "likelihood", "data") + (("inference",) if "inference" in doc else ()) + (("inducing_inputs",) if "inducing_inputs" in doc else ()))
    check(same, "%s round trips through to_wire exactly" % name)
    errs = list(validator.iter_errors(back))
    check(not errs, "%s to_wire output validates against the schema" % name)
    m2 = A.from_wire(back)
    check(abs(m2.log_likelihood() - m.log_likelihood()) < 1e-12, "%s rebuilt model has the same log likelihood" % name)

# Estimated-beta modes (CIP-0009): no GPy model realises them, so only reference() is exercised.
for name in ["ftc_linear_profiled", "ftc_linear_flat_prior", "ftc_linear_gaussian_prior", "ftc_linear_gaussian_prior_diag", "ftc_linear_quadratic_profiled"]:
    doc = load(root / "examples" / (name + ".yaml"))
    ref = doc["x-reference"]; tol = ref["tolerance"]
    r = A.reference(doc, ref["Xnew"])
    check(abs(r["log_likelihood"] - ref["log_likelihood"]) <= tol["log_likelihood"]
          and np.abs(np.array(r["mean"]) - ref["mean"]).max() <= tol["mean"]
          and np.abs(np.array(r["var"]) - ref["var"]).max() <= tol["var"],
          "%s reproduces its x-reference from GPy's K and noise" % name)
    raises(lambda: A.from_wire(doc), UnsupportedByFramework, "%s: no GPy model for estimated beta" % name)

# The bias/linear rewrite used for GPc and GPmat, checked against GPy's own bias and linear kernels.
from gp_scuderia.linear_basis import kernel_form
for name in ["ftc_linear_gaussian_prior_diag", "ftc_linear_intercept_fixed"]:
    doc = load(root / "examples" / (name + ".yaml")); ref = doc["x-reference"]; tol = ref["tolerance"]
    kf = kernel_form(doc)
    check("mean" not in kf or kf["mean"]["type"] == "mean.constant", "%s rewrite leaves no linear_basis mean" % name)
    r = A.reference(kf, ref["Xnew"])
    check(abs(r["log_likelihood"] - ref["log_likelihood"]) <= tol["log_likelihood"]
          and np.abs(np.array(r["mean"]) - ref["mean"]).max() <= tol["mean"]
          and np.abs(np.array(r["var"]) - ref["var"]).max() <= tol["var"],
          "%s kernel-form rewrite reproduces the x-reference in GPy" % name)

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
refuses(GPy.models.GPClassification(X, (Y > 0).astype(float), GPy.kern.RBF(1)), "non-Gaussian model with EP inference")
_b = load(root / "examples" / "laplace_bernoulli.yaml")
_p = load(root / "examples" / "laplace_poisson.yaml")
def _with(d, **kw):
    d = copy.deepcopy(d); d.update(kw); return d
_bad_labels = copy.deepcopy(_b); _bad_labels["data"]["Y"][0] = [2.0]
raises(lambda: A.from_wire(_bad_labels), ValueError, "Bernoulli labels outside {0, 1}")
_bad_counts = copy.deepcopy(_p); _bad_counts["data"]["Y"][0] = [1.5]
raises(lambda: A.from_wire(_bad_counts), ValueError, "non-integer Poisson counts")
check(bool(list(validator.iter_errors(_with(_b, inference="ep")))), "schema rejects inference: ep (reserved)")
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
