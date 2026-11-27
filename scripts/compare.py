"""Compare one framework against every golden's x-reference and record the result.

Usage: python scripts/compare.py <framework> [--json out.json] [--mode pinned|latest]
where <framework> is any frameworks/<name>.yaml.

Frameworks are run through the process-level JSON contract (python -m gp_scuderia.run, or the
Octave scripts): see schemas/task.schema.json and schemas/result.schema.json.

Each golden gets a status: match (within the golden's tolerance), mismatch, refused
(the adapter raised UnsupportedByFramework) or error. The registry's
expected outcome is derived from the golden's features and the framework file's declared
capabilities (scripts/frameworks.py); any other outcome is reported as unexpected and makes
the exit code non-zero.
"""
import argparse, importlib.metadata as md, json, pathlib, platform, sys, time, traceback
import numpy as np
import yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load
from gp_scuderia._common import UnsupportedByFramework
sys.path.insert(0, str(root / "scripts"))
import frameworks as F
from harness import launch, versions

frameworks = F.load_frameworks()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("framework", choices=sorted(frameworks))
    ap.add_argument("--json")
    ap.add_argument("--mode", default="local")
    a = ap.parse_args()
    fw = a.framework
    framework = frameworks[fw]
    rn = framework["runners"][0]
    extra = {}
    paths = sorted((root / "examples").glob("*.yaml"))
    tasks = [{"name": p.stem, "golden": str(p.with_suffix(".json"))} for p in paths]
    raw = launch(fw, rn, tasks, extra)
    expected_refusals = set()
    results, unexpected = [], 0
    for p in paths:
        name = p.stem
        doc = load(p); ref = doc["x-reference"]; tol = ref["tolerance"]
        why = F.missing(framework["_caps"], doc)
        if why:
            expected_refusals.add(name)
        rec = {"golden": name, "expected": "refused" if why else "match"}
        t0 = time.time()
        try:
            g = raw[name]
            if "refused" in g:
                raise UnsupportedByFramework(g["refused"])
            if "error" in g:
                raise RuntimeError(g["error"])
            dev = {"log_likelihood": abs(g["log_likelihood"] - ref["log_likelihood"]),
                   "mean": float(np.abs(np.array(g["mean"]) - ref["mean"]).max()),
                   "var": float(np.abs(np.array(g["var"]) - ref["var"]).max())}
            if "latent_mean" in ref and "latent_mean" in g:
                dev["latent_mean"] = float(np.abs(np.array(g["latent_mean"]) - ref["latent_mean"]).max())
                dev["latent_var"] = float(np.abs(np.array(g["latent_var"]) - ref["latent_var"]).max())
            lim = {"log_likelihood": tol["log_likelihood"], "mean": tol["mean"], "var": tol["var"],
                   "latent_mean": tol["mean"], "latent_var": tol["var"]}
            rec.update(deviation=dev, tolerance={k: lim[k] for k in dev},
                       status="match" if all(dev[k] <= lim[k] for k in dev) else "mismatch")
        except UnsupportedByFramework as e:
            rec.update(status="refused", reason=str(e))
        except Exception as e:
            rec.update(status="error", reason="%s: %s" % (type(e).__name__, e),
                       trace=traceback.format_exc()[-1500:])
        rec["seconds"] = round(time.time() - t0, 3)
        rec["ok"] = rec["status"] == rec["expected"]
        unexpected += (not rec["ok"])
        results.append(rec)
        dev = rec.get("deviation")
        print("%-5s %-26s %-8s %s" % ("ok" if rec["ok"] else "FAIL", name, rec["status"],
              ("max dev/tol = %.2g" % max(dev[k] / rec["tolerance"][k] for k in dev)) if dev else rec.get("reason", "")[:90]))
    doc = {"framework": fw, "mode": a.mode, "python": platform.python_version(),
           "platform": platform.platform(), "versions": {**versions(rn), **extra},
           "expected_refusals": sorted(expected_refusals), "results": results}
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=2) + "\n")
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
