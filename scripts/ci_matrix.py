"""Emit the CI job matrix from frameworks/*.yaml (CIP-0003).

Usage: python scripts/ci_matrix.py [--latest] [--github-output]
Entries are {framework, mode, kind, install?, env?, unsupported_check}; python, r and julia runners
carry an install list, octave runners an env variable. Frameworks with
ci.job set (bespoke jobs such as gpc) are left to their own job. Latest-mode entries are
only emitted with --latest (scheduled and manual runs).
"""
import argparse, json, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frameworks as F


def matrix(include_latest):
    out = []
    for name, fw in F.load_frameworks().items():
        ci = fw.get("ci", {})
        if "job" in ci:
            continue
        rn = fw["runners"][0]
        for mode in ci.get("modes", ["pinned"]):
            if mode == "latest" and not include_latest:
                continue
            e = {"framework": name, "mode": mode, "kind": rn["kind"],
                 "unsupported_check": bool(ci.get("unsupported_check")) and mode == "pinned"}
            if rn["kind"] in ("python", "r", "julia"):
                e["install"] = rn["install"][mode]
            else:
                e["env"] = rn["env"]
            out.append(e)
    return {"include": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--lines", action="store_true", help="pinned entries as: framework kind env-or-dash")
    ap.add_argument("--github-output", action="store_true")
    a = ap.parse_args()
    if a.lines:
        for e in matrix(False)["include"]:
            print(e["framework"], e["kind"], e.get("env", "-"))
        sys.exit(0)
    text = json.dumps(matrix(a.latest), separators=(",", ":"))
    if a.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("matrix=%s\n" % text)
    print(text)
