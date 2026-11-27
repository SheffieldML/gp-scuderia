"""Process-level runner for Python framework adapters (CIP-0003).

    python -m gp_scuderia.run --entry module:func --task task.json --out result.json

The task file lists goldens (schemas/task.schema.json); the result file holds one raw result
per task (schemas/result.schema.json). No tolerances are applied here. Each task is caught
individually: UnsupportedByFramework becomes {"refused": ...}, any other exception becomes
{"error": ..., "trace": ...}. A hard crash of the process is handled by the caller, which
retries the tasks one at a time.
"""
import argparse, importlib, json, pathlib, platform, time, traceback

import numpy as np

from ._common import UnsupportedByFramework
from .io import load


def _time(func, doc, Xnew, spec, cold):
    for _ in range(spec.get("warmup", 1)):
        func(doc, Xnew)
    ts = []
    for _ in range(spec["repeats"]):
        t0 = time.perf_counter(); func(doc, Xnew); ts.append(time.perf_counter() - t0)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return {"seconds_cold": cold, "seconds_median": float(med), "seconds_iqr": float(q3 - q1), "repeats": len(ts)}


def run_tasks(func, tasks):
    results = {}
    for t in tasks:
        try:
            doc = load(t["golden"])
            Xnew = doc["x-reference"]["Xnew"]
            t0 = time.perf_counter()
            g = func(doc, Xnew)          # the first call in this process for this task is the cold one
            _cold = time.perf_counter() - t0
            r = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in g.items()}
            if t.get("timing"):
                r["timing"] = _time(func, doc, Xnew, t["timing"], cold=_cold)
            results[t["name"]] = r
        except UnsupportedByFramework as e:
            results[t["name"]] = {"refused": str(e)}
        except Exception as e:
            results[t["name"]] = {"error": "%s: %s" % (type(e).__name__, e),
                                  "trace": traceback.format_exc()[-1500:]}
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    module, name = a.entry.split(":")
    func = getattr(importlib.import_module(module), name)
    task = json.loads(pathlib.Path(a.task).read_text())
    out = {"schema": "gp-scuderia-result/1", "runtime": {"python": platform.python_version()},
           "results": run_tasks(func, task["tasks"])}
    pathlib.Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
