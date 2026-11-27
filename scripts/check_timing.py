"""Check the timing machinery (CIP-0005) without any framework.

- grid and tier expansion: counts, unique ids, the pr tier is a subset and scheduled is everything
- the in-runner timer: the call count (one cold, warmup, repeats), and a stub of known duration is timed within bounds
- the schemas: timing is optional (v1 files still validate), a timed result validates, bad repeats are rejected
"""
import json, pathlib, sys, tempfile, time
import yaml, jsonschema

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
from benchmarks import expand_grid
from gp_scuderia import run as runner

fails = []


def check(ok, msg):
    print("%-5s %s" % ("ok" if ok else "FAIL", msg))
    if not ok:
        fails.append(msg)


b = expand_grid(yaml.safe_load((root / "benchmarks" / "timing_exact.yaml").read_text()))
ids = [c["id"] for c in b["cases"]]
check(len(ids) == 14 and len(set(ids)) == 14, "timing_exact expands to 14 unique cases (7 sizes x 2 dimensions)")
pr = [c["id"] for c in b["cases"] if "pr" in c["tiers"]]
check(sorted(pr) == ["n100_d2", "n1600_d2", "n400_d2"], "pr tier keeps the listed sizes in d=2 only: %s" % pr)
check(all("scheduled" in c["tiers"] for c in b["cases"]), "the scheduled tier is everything")
s = expand_grid(yaml.safe_load((root / "benchmarks" / "timing_sparse.yaml").read_text()))
check(len(s["cases"]) == 12 and {c["approximation"] for c in s["cases"]} == {"dtcvar", "fitc"}, "timing_sparse expands over n, M and approximation")

calls = []
def stub(doc, Xnew):
    calls.append(time.perf_counter()); time.sleep(0.02)
    return {"log_likelihood": -1.0, "mean": [0.0], "var": [1.0]}

tmp = pathlib.Path(tempfile.mkdtemp()); g = tmp / "g.json"
g.write_text(json.dumps({"x-reference": {"Xnew": [[0.0]]}}))
out = runner.run_tasks(stub, [{"name": "t", "golden": str(g), "timing": {"repeats": 5, "warmup": 1}},
                              {"name": "u", "golden": str(g)}])
tm = out["t"]["timing"]
check(len(calls) == 7 + 1, "one cold + 1 warmup + 5 timed calls, and one untimed task call (got %d)" % len(calls))
check(0.02 <= tm["seconds_median"] < 0.2 and tm["seconds_cold"] >= 0.02 and tm["repeats"] == 5 and tm["seconds_iqr"] >= 0,
      "stub of 20 ms is timed as %.4g s (cold %.4g s)" % (tm["seconds_median"], tm["seconds_cold"]))
check("timing" not in out["u"], "a task without timing carries no timing")

task = json.loads((root / "schemas" / "task.schema.json").read_text())
res = json.loads((root / "schemas" / "result.schema.json").read_text())
jsonschema.validate({"schema": "gp-scuderia-task/1", "tasks": [{"name": "a", "golden": "g.json"}]}, task)
jsonschema.validate({"schema": "gp-scuderia-task/1", "tasks": [{"name": "a", "golden": "g.json", "timing": {"repeats": 3}}]}, task)
try:
    jsonschema.validate({"schema": "gp-scuderia-task/1", "tasks": [{"name": "a", "golden": "g.json", "timing": {"repeats": 0}}]}, task)
    check(False, "repeats 0 is rejected")
except jsonschema.ValidationError:
    check(True, "v1 task files validate; a timed task validates; repeats 0 is rejected")
jsonschema.validate({"schema": "gp-scuderia-result/1", "results": out}, res)
check(True, "a timed result validates against the result schema")
sys.exit(1 if fails else 0)
