"""Check the process-level runner contract with a stub framework (CIP-0003).

The stub returns each golden's own reference, so it needs no framework installed. It crashes
the whole process on one golden, raises on another and refuses a third. We check that the
task and result files satisfy their schemas, that the crash is isolated to its own task, and
that the other tasks still come back intact.
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
res = compare.launch("stub", {"kind": "python", "entry": "stub_runner:reference"}, tasks, {})
jsonschema.validate({"schema": "gp-scuderia-result/1", "results": res}, json.loads((root / "schemas/result.schema.json").read_text()))

bad = 0
def check(label, ok):
    global bad
    print(("ok   " if ok else "FAIL ") + label)
    bad += not ok
check("crash isolated to its own task", "crashed" in res[crash].get("error", ""))
check("exception becomes an error result", "stub failure" in res[raise_].get("error", ""))
check("refusal passes through", res[refuse].get("refused") == "stub refusal")
others = [n for n in goldens if n not in (crash, raise_, refuse)]
check("other %d tasks intact after the crash" % len(others),
      all(res[n]["log_likelihood"] == goldens[n]["x-reference"]["log_likelihood"] for n in others))
sys.exit(1 if bad else 0)
