"""Compare one framework against every golden's x-reference and record the result.

Usage: python scripts/compare.py <gpy|gpflow|gpytorch|gpjax|gpc|gpml|gpmat> [--json out.json] [--mode pinned|latest]

Each golden gets a status: match (within the golden's tolerance), mismatch, refused
(the adapter raised UnsupportedByFramework) or error. The registry's
framework_verification.<fw>.refuses lists what is expected to be refused; any other
outcome than the expected one is reported as unexpected and makes the exit code non-zero.
"""
import argparse, importlib, importlib.metadata as md, json, pathlib, platform, sys, time, traceback
import numpy as np
import yaml

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python"))
from gp_scuderia import load
from gp_scuderia._common import UnsupportedByFramework

PACKAGES = {
    "gpy": ["GPy", "numpy", "scipy"],
    "gpflow": ["gpflow", "tensorflow", "tensorflow-probability", "numpy"],
    "gpytorch": ["gpytorch", "torch", "numpy"],
    "gpjax": ["gpjax", "jax", "jaxlib", "numpyro", "numpy"],
    "gpc": ["gpc", "numpy"],
    "gpml": [],   # not Python packages: Octave and toolbox versions are recorded from the run
    "gpmat": [],
}


def versions(fw):
    out = {}
    for p in PACKAGES[fw]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


OCTAVE_RUNNERS = {"gpml": ("GPML", "octave/gpml/runGpml.m"), "gpmat": ("GPMAT", "octave/gpmat/runGpmat.m")}


def octave_runner(fw, extra):
    """Run the framework's Octave script once and serve its raw results (env var names the checkout)."""
    import os, subprocess, tempfile
    var, script = OCTAVE_RUNNERS[fw]
    if not os.environ.get(var):
        raise SystemExit("Set %s to a %s checkout." % (var, fw))
    out = pathlib.Path(tempfile.mkdtemp()) / "gpml.json"
    env = dict(os.environ, GPSCUDERIA_ROOT=str(root), GPSCUDERIA_OUT=str(out))
    subprocess.run(["octave-cli", "--no-gui", str(root / script)], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    raw = json.loads(out.read_text())
    extra.update(octave=raw["octave"], **{fw: raw[fw]})

    def run(doc, Xnew, name):
        r = raw["results"][name]
        if "refused" in r:
            raise UnsupportedByFramework(r["refused"])
        if "error" in r:
            raise RuntimeError(r["error"])
        return r
    return run


def runner(fw):
    if fw == "gpc":
        from gpc import interchange as I
        return lambda doc, Xnew: I.reference(I.from_wire(doc), Xnew)
    A = importlib.import_module("gp_scuderia.%s_adapter" % fw)
    return A.reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("framework", choices=sorted(PACKAGES))
    ap.add_argument("--json")
    ap.add_argument("--mode", default="local")
    a = ap.parse_args()
    fw = a.framework
    registry = yaml.safe_load((root / "registry/types.yaml").read_text())
    expected_refusals = set(registry.get("framework_verification", {}).get(fw, {}).get("refuses", []))
    extra = {}
    if fw in OCTAVE_RUNNERS:
        run = octave_runner(fw, extra)
    else:
        base = runner(fw)
        run = lambda doc, Xnew, name: base(doc, Xnew)
    results, unexpected = [], 0
    for p in sorted((root / "examples").glob("*.yaml")):
        name = p.stem
        doc = load(p); ref = doc["x-reference"]; tol = ref["tolerance"]
        rec = {"golden": name, "expected": "refused" if name in expected_refusals else "match"}
        t0 = time.time()
        try:
            g = run(doc, ref["Xnew"], name)
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
           "platform": platform.platform(), "versions": {**versions(fw), **extra},
           "registry_expected_refusals": sorted(expected_refusals), "results": results}
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=2) + "\n")
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
