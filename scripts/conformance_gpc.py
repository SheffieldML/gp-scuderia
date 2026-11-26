"""Conformance of the GPc adapter against the goldens' x-reference (GPy oracle).

Run: PYTHONPATH=<GPc>/python:<gp-scuderia>/python /opt/anaconda3/bin/python scripts/conformance_gpc.py
Reports (rather than hides) documented divergences.
"""
import pathlib, sys, copy
import numpy as np
from gp_scuderia import load
from gpc import interchange as I

root = pathlib.Path(__file__).resolve().parents[1]
bad = 0

def check(cond, msg):
    global bad
    print(("ok   " if cond else "FAIL ") + msg)
    bad += (not cond)

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52",
             "ftc_standardize", "dtcvar_mean_constant", "ftc_icm", "ftc_lcm"]:
    doc = load(root / "examples" / (name + ".yaml"))
    ref = doc["x-reference"]; tol = ref["tolerance"]
    m = I.from_wire(doc)
    g = I.reference(m, ref["Xnew"])
    dl = abs(g["log_likelihood"] - ref["log_likelihood"])
    dm = np.abs(np.array(g["mean"]) - ref["mean"]).max()
    dv = np.abs(np.array(g["var"]) - ref["var"]).max()
    check(dl <= tol["log_likelihood"] and dm <= tol["mean"] and dv <= tol["var"],
          "%s matches GPy: dLL=%.2g dmean=%.2g dvar=%.2g" % (name, dl, dm, dv))
    back = I.to_wire(m, doc)
    def leaves(n):
        return [x for p in n["parts"] for x in leaves(p)] if "parts" in n else [n]
    lb, lo = leaves(back["kernel"]), leaves(doc["kernel"])
    def same_leaf(a, b):
        if a["type"] != b["type"] or a.get("active_dims") != b.get("active_dims"):
            return False
        if a["type"] == "kern.coregionalize":
            return (np.allclose(a["W"], b["W"], rtol=1e-12)
                    and np.allclose(a["kappa"], b["kappa"], rtol=1e-12))
        return (np.isclose(a["variance"], b["variance"], rtol=1e-12)
                and np.allclose(a.get("lengthscale", 0), b.get("lengthscale", 0), rtol=1e-12))
    check(len(lb) == len(lo) and all(same_leaf(a, b) for a, b in zip(lb, lo)),
          "%s kernel round-trips through GPc storage" % name)
    check(np.allclose(back["likelihood"]["variance"], doc["likelihood"]["variance"], rtol=1e-12),
          "%s likelihood variance round-trips" % name)

# Non-Gaussian: GPc's own Laplace (CIP-0033/CIP-003A) against GPy and the independent numpy check.
for name in ["laplace_bernoulli", "laplace_poisson"]:
    doc = load(root / "examples" / (name + ".yaml"))
    ref = doc["x-reference"]; tol = ref["tolerance"]
    m = I.from_wire(doc)
    g = I.reference(m, ref["Xnew"])
    d = [abs(g["log_likelihood"] - ref["log_likelihood"]),
         np.abs(np.array(g["mean"]) - ref["mean"]).max(), np.abs(np.array(g["var"]) - ref["var"]).max(),
         np.abs(np.array(g["latent_mean"]) - ref["latent_mean"]).max(),
         np.abs(np.array(g["latent_var"]) - ref["latent_var"]).max()]
    check(d[0] <= tol["log_likelihood"] and max(d[1], d[3]) <= tol["mean"] and max(d[2], d[4]) <= tol["var"],
          "%s matches GPy: dLL=%.2g dmean=%.2g dvar=%.2g dlatent=%.2g/%.2g" % (name, d[0], d[1], d[2], d[3], d[4]))
    back = I.to_wire(m, doc)
    check(back["likelihood"] == doc["likelihood"] and back["inference"] == "laplace"
          and np.isclose(back["kernel"]["variance"], doc["kernel"]["variance"], rtol=1e-12),
          "%s round-trips through GPc storage" % name)
_bad = load(root / "examples" / "laplace_bernoulli.yaml"); _bad["data"]["Y"][0] = [2.0]
try:
    I.from_wire(_bad); check(False, "rejects Bernoulli label 2")
except Exception as e:
    check(True, "rejects Bernoulli label 2 (%s)" % type(e).__name__)
_bad = load(root / "examples" / "laplace_poisson.yaml"); _bad["data"]["Y"][0] = [1.5]
try:
    I.from_wire(_bad); check(False, "rejects fractional Poisson count")
except Exception as e:
    check(True, "rejects fractional Poisson count (%s)" % type(e).__name__)

# CIP-0027 default is unchanged: without the opt-in a trainable White owns the noise.
doc = load(root / "examples" / "ftc_rbf_white.yaml")
m = I.from_wire(doc); m.set_allow_white_and_noise(False)
check(abs(-m.objective() - doc["x-reference"]["log_likelihood"]) > 0.1,
      "default (CIP-0027) noise ownership preserved without opt-in")

# Negative tests: refuse rather than guess.
base = load(root / "examples" / "dtcvar_ard_data.yaml")
def rejects(mut, label):
    d = copy.deepcopy(base); mut(d)
    try:
        I.from_wire(d); check(False, "rejects " + label)
    except (I.InterchangeError, KeyError, ValueError, RuntimeError) as e:
        check(True, "rejects %s (%s)" % (label, type(e).__name__))
rejects(lambda d: d["kernel"]["parts"][0].update(type="kern.nonesuch"), "unknown kernel type")
rejects(lambda d: d["data"].update(X=[r[:1] for r in d["data"]["X"]]), "data width != input_dim")
rejects(lambda d: d["kernel"]["parts"][0].update(lengthscale=[0.8]), "ARD lengthscale length != input_dim")
rejects(lambda d: d.update(schema_version="9.9"), "unsupported schema_version")
rejects(lambda d: d.pop("data"), "missing data")
sys.exit(1 if bad else 0)
