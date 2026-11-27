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


def _tolist(v):
    return v.tolist() if hasattr(v, "tolist") else v


def run_tasks(func, tasks, grad=None, bound=None, psi=None):
    """grad is the framework's gradient(doc) -> {wire path: dLL/dparameter} or None if it has none.
    bound(doc) -> {bound_without_kl, kl} and psi(doc) -> {psi0, psi1, psi2} are the latent-variable
    operations of contract /3 (CIP-0006); like gradient they are requested on their own."""
    results = {}
    for t in tasks:
        try:
            doc = load(t["golden"])
            Xnew = doc.get("x-reference", {}).get("Xnew")
            ops = t.get("operations")
            if ops == ["gradient"]:
                if grad is None:
                    raise UnsupportedByFramework("this adapter provides no gradient")
                g = grad(doc)
                results[t["name"]] = {"gradient": {k: (np.asarray(v, float).tolist() if np.ndim(v) else float(v))
                                                   for k, v in g.items()}}
                continue
            if ops in (["bound"], ["psi"]):
                fn = {"bound": bound, "psi": psi}[ops[0]]
                if fn is None:
                    raise UnsupportedByFramework("this adapter provides no %s operation" % ops[0])
                results[t["name"]] = {ops[0]: {k: (float(v) if np.ndim(v) == 0 else _tolist(np.asarray(v, float)))
                                               for k, v in fn(doc).items()}}
                continue
            if ops and set(ops) - {"log_likelihood", "predict"}:
                raise UnsupportedByFramework("operations %s: gradient, bound and psi must each be requested on their own" % ops)
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
    mod = importlib.import_module(module)
    func, grad = getattr(mod, name), getattr(mod, "gradient", None)
    bound, psi = getattr(mod, "bound", None), getattr(mod, "psi", None)
    task = json.loads(pathlib.Path(a.task).read_text())
    out = {"schema": "gp-scuderia-result/3", "runtime": {"python": platform.python_version()},
           "results": run_tasks(func, task["tasks"], grad, bound, psi)}
    pathlib.Path(a.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
