"""Shared runner plumbing for compare.py and benchmarks.py (CIP-0003)."""
import importlib.metadata as md, json, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]


def versions(runner):
    out = {}
    for p in runner.get("versions", []):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


THREAD_VARS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
               "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
               "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}


def launch(fw, runner, tasks, extra, threads=None):
    """Run tasks through the process-level contract (task file in, result file out).

    A crash of the whole process triggers one isolated retry per task, so one bad golden
    cannot hide the others; a task that still crashes is reported as an error.
    """
    import os, subprocess, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    env = dict(os.environ, GPSCUDERIA_ROOT=str(root),
               PYTHONPATH=os.pathsep.join(filter(None, [str(root / "python"), os.environ.get("PYTHONPATH")])))
    if threads == 1:        # best effort: the variables the numerical libraries in use are known to honour
        env.update(THREAD_VARS)
    if runner["kind"] == "octave" and not os.environ.get(runner["env"]):
        raise SystemExit("Set %s to a %s checkout." % (runner["env"], fw))

    def once(batch, tag):
        task, out = tmp / (tag + "-task.json"), tmp / (tag + "-result.json")
        # the lowest contract version that covers the batch, so older runners keep working
        ops = [o for t in batch for o in t.get("operations", [])]
        ver = 3 if {"bound", "psi"} & set(ops) else 2 if any("operations" in t for t in batch) else 1
        task.write_text(json.dumps({"schema": "gp-scuderia-task/%d" % ver, "tasks": batch}))
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
