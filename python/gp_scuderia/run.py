"""Process-level runner for Python framework adapters (CIP-0003).

    python -m gp_scuderia.run --entry module:func --task task.json --out result.json

The task file lists goldens (schemas/task.schema.json); the result file holds one raw result
per task (schemas/result.schema.json). No tolerances are applied here. Each task is caught
individually: UnsupportedByFramework becomes {"refused": ...}, any other exception becomes
{"error": ..., "trace": ...}. A hard crash of the process is handled by the caller, which
retries the tasks one at a time.
"""
import argparse, importlib, json, pathlib, platform, traceback

from ._common import UnsupportedByFramework
from .io import load


def run_tasks(func, tasks):
    results = {}
    for t in tasks:
        try:
            doc = load(t["golden"])
            g = func(doc, doc["x-reference"]["Xnew"])
            results[t["name"]] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in g.items()}
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
