"""Check the process-level runner contract with a stub framework (CIP-0003, CIP-0007).

The stub returns each golden's own reference, so it needs no framework installed. It crashes
the whole process on one golden, raises on another and refuses a third. We check that the
task and result files satisfy their schemas, that the crash is isolated to its own task, and
that the other tasks still come back intact. The same stub exists for each runner kind
(python, and r when Rscript is available), so the check covers every launch path.
"""
import json, pathlib, sys, tempfile
import jsonschema

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
import compare
from gp_scuderia import load
from gp_scuderia._common import UnsupportedByFramework

stub = pathlib.Path(tempfile.mkdtemp()) / "stub_runner.py"
goldens = {p.stem: load(p) for p in sorted((root / "examples").glob("*.yaml"))}
# Behaviour is selected by golden log-likelihood, which is unique per golden.
crash, raise_, refuse = "ftc_icm", "ftc_lcm", "ftc_standardize"
stub.write_text('''
import os
from gp_scuderia._common import UnsupportedByFramework
LL = %r
def reference(doc, Xnew):
    ll = doc["x-reference"]["log_likelihood"]
    if ll == LL["crash"]: os._exit(3)
    if ll == LL["raise"]: raise RuntimeError("stub failure")
    if ll == LL["refuse"]: raise UnsupportedByFramework("stub refusal")
    ref = doc["x-reference"]
    return {k: ref[k] for k in ("log_likelihood", "mean", "var")}
''' % {"crash": goldens[crash]["x-reference"]["log_likelihood"],
       "raise": goldens[raise_]["x-reference"]["log_likelihood"],
       "refuse": goldens[refuse]["x-reference"]["log_likelihood"]})
sys.path.insert(0, str(stub.parent))
import os
os.environ["PYTHONPATH"] = str(stub.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")

tasks = [{"name": n, "golden": str(root / "examples" / (n + ".json"))} for n in goldens]
jsonschema.validate({"schema": "gp-scuderia-task/1", "tasks": tasks}, json.loads((root / "schemas/task.schema.json").read_text()))

bad = 0
def check(label, ok):
    global bad
    print(("ok   " if ok else "FAIL ") + label)
    bad += not ok

def verify(kind, res, exact):
    res_schema = json.loads((root / "schemas/result.schema.json").read_text())
    jsonschema.validate({"schema": "gp-scuderia-result/1", "results": res}, res_schema)
    check("[%s] crash isolated to its own task" % kind, "crashed" in res[crash].get("error", ""))
    check("[%s] exception becomes an error result" % kind, "stub failure" in res[raise_].get("error", ""))
    check("[%s] refusal passes through" % kind, res[refuse].get("refused") == "stub refusal")
    others = [n for n in goldens if n not in (crash, raise_, refuse)]
    ref = lambda n: goldens[n]["x-reference"]["log_likelihood"]
    same = (lambda a, b: a == b) if exact else (lambda a, b: abs(a - b) <= 1e-12 * max(1.0, abs(b)))
    check("[%s] other %d tasks intact after the crash" % (kind, len(others)),
          all(same(res[n]["log_likelihood"], ref(n)) for n in others))

verify("python", compare.launch("stub", {"kind": "python", "entry": "stub_runner:reference"}, tasks, {}), exact=True)

import shutil
if shutil.which("Rscript"):
    # The R stub reads the same goldens through jsonlite, which keeps 15 significant digits.
    ll = lambda n: repr(goldens[n]["x-reference"]["log_likelihood"])
    os.environ.update(GPSCUDERIA_STUB_CRASH=ll(crash), GPSCUDERIA_STUB_RAISE=ll(raise_), GPSCUDERIA_STUB_REFUSE=ll(refuse))
    extra = {}
    runner = {"kind": "r", "script": "R/stub/runStub.R"}
    verify("r", compare.launch("stub", runner, tasks, {}), exact=False)
    clean = [t for t in tasks if t["name"] not in (crash, raise_, refuse)]   # a whole-batch success records the runtime
    compare.launch("stub", runner, clean, extra)
    check("[r] runtime recorded in the result", "r" in extra and "stub" in extra)
else:
    print("skip R runner check: Rscript not on PATH")

julia = os.environ.get("JULIA", "julia")
if shutil.which(julia):
    ll = lambda n: repr(goldens[n]["x-reference"]["log_likelihood"])
    os.environ.update(GPSCUDERIA_STUB_CRASH=ll(crash), GPSCUDERIA_STUB_RAISE=ll(raise_), GPSCUDERIA_STUB_REFUSE=ll(refuse))
    extra = {}
    runner = {"kind": "julia", "script": "julia/stub/runStub.jl"}
    verify("julia", compare.launch("stub", runner, tasks, {}), exact=False)
    clean = [t for t in tasks if t["name"] not in (crash, raise_, refuse)]
    compare.launch("stub", runner, clean, extra)
    check("[julia] runtime recorded in the result", "julia" in extra and "stub" in extra)
else:
    print("skip Julia runner check: julia not on PATH")
sys.exit(1 if bad else 0)
