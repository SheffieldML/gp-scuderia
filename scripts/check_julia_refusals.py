"""Julia runners refuse what they cannot realise exactly, and never substitute (CIP-000A).

Usage: python scripts/check_julia_refusals.py juliagps
Each case edits a golden into something outside the package's exact-Gaussian slice and checks
that the runner answers `refused`, not a number and not an error. Cases that must run
(controls) guard against a runner that refuses everything.
"""
import copy, json, pathlib, sys, tempfile
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
import frameworks as F
from harness import launch

fw = sys.argv[1]
framework = F.load_frameworks()[fw]
base = json.loads((root / "examples/ftc_standardize.json").read_text())
ref = base["x-reference"]
train = [r[0] for r in base["data"]["X"]]


def edit(**kw):
    d = copy.deepcopy(base)
    d.update(kw)
    return d


def kern(**kw):
    return dict(base["kernel"], **kw)


def at(xnew, **kw):
    d = edit(**kw)
    d["x-reference"]["Xnew"] = xnew
    return d


white = {"type": "kern.add", "parts": [base["kernel"], {"type": "kern.white", "variance": 0.1}]}
cases = {
    "control": (edit(), False),
    "ARD control": (edit(kernel=kern(ARD=True, lengthscale=[0.8])), False),
    "white kernel control (test input off the training set)": (at([[train[0] + 0.01]], kernel=white), False),
    "approximation dtc": (edit(approximation="dtc"), True),
    "approximation pitc": (edit(approximation="pitc"), True),
    "linear kernel": (edit(kernel={"type": "kern.linear", "variances": 1.0}), True),
    "active_dims": (edit(kernel=kern(active_dims=[0])), True),
    "white kernel at a training input": (at([[train[0]]], kernel=white), True),
    "mean with normalizer": (edit(mean={"type": "mean.constant", "value": 1.0}), True),
    "laplace inference": (edit(inference="laplace"), True),
    "bernoulli likelihood": (edit(likelihood={"type": "likelihoods.bernoulli"}), True),
}

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
