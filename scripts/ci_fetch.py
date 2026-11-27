"""Fetch a framework's source as declared in frameworks/<name>.yaml (runner `fetch:`).

Usage: python scripts/ci_fetch.py <framework> <dest-dir>
Prints the checkout path and, when GITHUB_ENV is set, exports <ENV> and <ENV>_VERSION
(ENV is the runner's `env:`). Archives are verified against their sha256; git sources are
checked out at the pinned revision.
"""
import hashlib, os, pathlib, subprocess, sys, tempfile, urllib.request, zipfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frameworks as F

fw, dest = sys.argv[1], pathlib.Path(sys.argv[2]).resolve()
rn = F.load_frameworks()[fw]["runners"][0]
f = rn["fetch"]
dest.mkdir(parents=True, exist_ok=True)
if "git" in f:
    subprocess.run(["git", "clone", "-q", f["git"], str(dest / fw)], check=True)
    subprocess.run(["git", "-C", str(dest / fw), "checkout", "-q", f["rev"]], check=True)
    path, version = dest / fw, f["rev"][:7]
else:
    data = urllib.request.urlopen(f["url"]).read()
    if hashlib.sha256(data).hexdigest() != f["sha256"]:
        sys.exit("sha256 mismatch for %s" % f["url"])
    with tempfile.NamedTemporaryFile(suffix=".zip") as t:
        t.write(data); t.flush()
        zipfile.ZipFile(t.name).extractall(dest / fw)
    inner = [p for p in (dest / fw).iterdir() if p.is_dir()]
    path, version = (inner[0] if len(inner) == 1 else dest / fw), str(f["version"])
print(path)
if os.environ.get("GITHUB_ENV"):
    with open(os.environ["GITHUB_ENV"], "a") as g:
        g.write("%s=%s\n%s_VERSION=%s\n" % (rn["env"], path, rn["env"], version))
