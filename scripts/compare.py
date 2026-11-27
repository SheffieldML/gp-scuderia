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

frameworks = F.load_frameworks()

def versions(runner):
    out = {}
    for p in runner.get("versions", []):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def launch(fw, runner, tasks, extra):
    """Run tasks through the process-level contract (task file in, result file out).

    A crash of the whole process triggers one isolated retry per task, so one bad golden
    cannot hide the others; a task that still crashes is reported as an error.
    """
    import os, subprocess, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    env = dict(os.environ, GPSCUDERIA_ROOT=str(root),
               PYTHONPATH=os.pathsep.join(filter(None, [str(root / "python"), os.environ.get("PYTHONPATH")])))
    if runner["kind"] == "octave" and not os.environ.get(runner["env"]):
        raise SystemExit("Set %s to a %s checkout." % (runner["env"], fw))

    def once(batch, tag):
        task, out = tmp / (tag + "-task.json"), tmp / (tag + "-result.json")
        task.write_text(json.dumps({"schema": "gp-scuderia-task/1", "tasks": batch}))
        if runner["kind"] == "octave":
            cmd = ["octave-cli", "--no-gui", str(root / runner["script"])]
            env.update(GPSCUDERIA_TASK=str(task), GPSCUDERIA_OUT=str(out))
        else:
            cmd = [sys.executable, "-m", "gp_scuderia.run", "--entry", runner["entry"],
                   "--task", str(task), "--out", str(out)]
        p = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0 or not out.exists():
            return None, "runner crashed (rc=%s): %s" % (p.returncode, p.stderr.strip()[-300:])
        return json.loads(out.read_text()), None

    raw, why = once(tasks, "batch")
    if raw is None:
        results = {}
        for t in tasks:
            r, why = once([t], t["name"])
            results[t["name"]] = r["results"][t["name"]] if r else {"error": why}
        return results
    for k in ("octave", fw):
        if k in raw:
            extra[k] = raw[k]
    return raw["results"]


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
