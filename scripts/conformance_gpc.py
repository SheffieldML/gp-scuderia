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

for name in ["ftc_rbf_white", "dtcvar_ard_data", "ftc_prod_matern52", "fitc_matern52"]:
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
    check(len(lb) == len(lo) and all(
              a["type"] == b["type"] and np.isclose(a["variance"], b["variance"], rtol=1e-12)
              and np.allclose(a.get("lengthscale", 0), b.get("lengthscale", 0), rtol=1e-12)
              for a, b in zip(lb, lo)),
          "%s kernel round-trips through GPc storage" % name)
    check(np.isclose(back["likelihood"]["variance"], doc["likelihood"]["variance"], rtol=1e-12),
          "%s likelihood variance round-trips" % name)

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
