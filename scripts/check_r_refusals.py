"""R runners refuse what they cannot realise exactly, and never substitute (CIP-0007).

Usage: python scripts/check_r_refusals.py kergp|dicekriging
Each case edits a golden into something outside the package's exact-Gaussian slice (sparse
bounds that have no golden, a linear kernel, ARD, active dimensions, an unsupported
kernel combination) and checks that the runner answers `refused`, not a number and not an
error. A case that must run (control) guards against a runner that refuses everything.
"""
import copy, json, pathlib, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
import frameworks as F
from harness import launch

fw = sys.argv[1]
framework = F.load_frameworks()[fw]
base = json.loads((root / "examples/ftc_standardize.json").read_text())

def edit(**kw):
    d = copy.deepcopy(base)
    for k, v in kw.items():
        d[k] = v
    return d

def kern(**kw):
    return dict(base["kernel"], **kw)

cases = {
    "control": (edit(), False),
    "approximation dtc": (edit(approximation="dtc"), True),
    "approximation pitc": (edit(approximation="pitc"), True),
    "linear kernel (trend)": (edit(kernel={"type": "kern.linear", "variances": 1.0}), True),
    "ARD": (edit(kernel=kern(ARD=True, lengthscale=[0.8])), True),
    "active_dims": (edit(kernel=kern(active_dims=[0])), True),
    "white kernel": (edit(kernel={"type": "kern.white", "variance": 0.1}), True),
    "mean with normalizer": (edit(mean={"type": "mean.constant", "value": 1.0}), True),
    "laplace inference": (edit(inference="laplace"), True),
}
# Compound kernels are outside DiceKriging only; kergp realises them.
if fw == "dicekriging":
    cases["isotropic Matern in two dimensions"] = (edit(input_dim=2, kernel=kern(type="kern.matern52"), data={"X": [[0.1, 0.2], [0.5, 0.9], [0.8, 0.3]], "Y": [[0.0], [1.0], [0.5]]}, **({})), True)
    cases["add of two kernels"] = (edit(kernel={"type": "kern.add", "parts": [base["kernel"], base["kernel"]]}), True)

tmp = pathlib.Path(tempfile.mkdtemp())
tasks = []
for i, (name, (doc, _)) in enumerate(cases.items()):
    p = tmp / ("case%d.json" % i); p.write_text(json.dumps(doc))
    tasks.append({"name": name, "golden": str(p)})
res = launch(fw, framework["runners"][0], tasks, {})
bad = 0
for name, (_, must_refuse) in cases.items():
    r = res[name]
    ok = ("refused" in r) if must_refuse else ("log_likelihood" in r)
    print("%s %s %s: %s" % ("ok  " if ok else "FAIL", fw, name, r.get("refused") or r.get("error") or "ran"))
    bad += not ok
sys.exit(1 if bad else 0)
