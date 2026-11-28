"""Round trip between GPy and juliagps for an RBF Gaussian regression (CIP-000A).

Run with the GPy interpreter after the Julia environment is installed:
    .venv-gpy/bin/python scripts/roundtrip_juliagps.py

Direction 1: GPy builds the model and writes the wire document; the juliagps runner reads it and
must reproduce GPy's log likelihood and predictions.
Direction 2: Julia builds the model natively and writes the wire document; GPy reads it and must
reproduce AbstractGPs' log likelihood and predictions.
"""
import json, os, pathlib, subprocess, sys, tempfile
import numpy as np
import GPy
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
import frameworks as F
from harness import launch
from gp_scuderia import gpy_adapter as A

bad = 0
def check(cond, msg):
    global bad
    print(("ok   " if cond else "FAIL ") + msg); bad += (not cond)

tmp = pathlib.Path(tempfile.mkdtemp())
Xnew = [[-1.5], [0.5], [3.0]]
TOL = 1e-6  # GPy and AbstractGPs differ at ~1e-7 (jitter and Cholesky details)

# Direction 1: GPy -> juliagps
rs = np.random.RandomState(3)
X = np.sort(rs.uniform(-2, 2.5, (8, 1)), axis=0); Y = np.sin(2 * X) + 0.1 * rs.randn(8, 1)
m = GPy.models.GPRegression(X, Y, GPy.kern.RBF(1, variance=1.3, lengthscale=0.7))
m.Gaussian_noise.variance = 0.05
doc = A.to_wire(m)
doc["x-reference"] = dict(A.reference(doc, Xnew), Xnew=Xnew)
p = tmp / "gpy_to_julia.json"; p.write_text(json.dumps(doc))
framework = F.load_frameworks()["juliagps"]
out = launch("juliagps", framework["runners"][0], [{"name": "rt", "golden": str(p)}], {})["rt"]
ref = doc["x-reference"]
check("refused" not in out and "error" not in out, "juliagps runs the GPy-written document (%s)" % {k: out[k] for k in out if k in ("refused", "error")})
if "log_likelihood" in out:
    check(abs(out["log_likelihood"] - ref["log_likelihood"]) < TOL, "GPy -> juliagps: log likelihood agrees")
    check(np.abs(np.array(out["mean"]) - ref["mean"]).max() < TOL, "GPy -> juliagps: predictive mean agrees")
    check(np.abs(np.array(out["var"]) - ref["var"]).max() < TOL, "GPy -> juliagps: predictive variance agrees")

# Direction 2: juliagps -> GPy
julia = os.environ.get("JULIA", "julia")
q = tmp / "julia_to_gpy.json"
subprocess.run([julia, "--project=" + str(root / "julia"), str(root / "julia/juliagps/exportModel.jl"), str(q)], check=True)
jdoc = json.loads(q.read_text())
jref = jdoc["x-reference"]
back = A.reference(jdoc, jref["Xnew"])
check(abs(back["log_likelihood"] - jref["log_likelihood"]) < TOL, "juliagps -> GPy: log likelihood agrees")
check(np.abs(np.array(back["mean"]) - jref["mean"]).max() < TOL, "juliagps -> GPy: predictive mean agrees")
check(np.abs(np.array(back["var"]) - jref["var"]).max() < TOL, "juliagps -> GPy: predictive variance agrees")
sys.exit(1 if bad else 0)
