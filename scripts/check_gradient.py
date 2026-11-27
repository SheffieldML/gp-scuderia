"""Check the gradient machinery (CIP-0005), with GPy only.

1. Every case of the gradient benchmark generates, and the oracle's analytic gradient agrees
   with central differences (the generator enforces this and records the worst relative gap).
2. The comparison accepts the oracle and rejects the typical conversion mistake: a framework
   that optimises in log space and forgets the chain rule reports dL/d(log theta) = theta dL/d theta.
3. Task and result files of contract /1 still validate against the /2 schemas, and /2 files
   with `operations` validate.
"""
import json, pathlib, sys
import jsonschema

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
import benchmarks as B
from gp_scuderia import gpy_adapter
from gp_scuderia.params import wire_paths

bench = next(b for b in B.benchmarks() if b["name"] == "gradient_agreement")
tol = bench["tolerance"]["gradient"]
status, wrong_accepted = 0, 0
for i, case in enumerate(bench["cases"]):
    doc, Xnew = B.make_model(bench, case, i)
    ref, fd_gap = B.oracle_gradient(doc, Xnew)
    ok_dev, _ = B.gradient_deviation(ref, ref, tol)
    wrong, values = {}, dict(wire_paths(doc))
    for k in ref:   # the log-space gradient: multiply by the parameter value
        val = values[k]
        wrong[k] = [g * v for g, v in zip(ref[k], val)] if isinstance(ref[k], list) else ref[k] * val
    bad_dev, _ = B.gradient_deviation(wrong, ref, tol)
    good = ok_dev == 0 and fd_gap < 1e-4 and bad_dev > 1
    # a parameter equal to 1 is its own log-space gradient; the case is still caught by the others
    print("%-5s %-24s fd gap %.1e, wrong-conversion deviation %.3g" % ("ok" if good else "FAIL", case["id"], fd_gap, bad_dev))
    status |= not good

task_schema = json.loads((root / "schemas/task.schema.json").read_text())
result_schema = json.loads((root / "schemas/result.schema.json").read_text())
t1 = {"schema": "gp-scuderia-task/1", "tasks": [{"name": "a", "golden": "a.json"}]}
t2 = {"schema": "gp-scuderia-task/2", "tasks": [{"name": "a", "golden": "a.json", "operations": ["gradient"]}]}
r2 = {"schema": "gp-scuderia-result/2", "results": {"a": {"gradient": {"kernel.variance": 1.0, "kernel.lengthscale": [1.0, 2.0]}}}}
for doc, sch, what in ((t1, task_schema, "task /1"), (t2, task_schema, "task /2"), (r2, result_schema, "result /2")):
    try:
        jsonschema.validate(doc, sch); print("ok    %s validates" % what)
    except jsonschema.ValidationError as e:
        print("FAIL  %s: %s" % (what, e.message)); status = 1
sys.exit(status)
