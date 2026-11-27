"""Check frameworks/*.yaml: valid against the registry, derived expectations match the
snapshot in tests/expected_refusals.yaml, and a wrong declaration flips the expectation
in each direction. Needs only pyyaml."""
import copy, pathlib, sys
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frameworks as F

root = F.root
fws = F.load_frameworks()
goldens = {p.stem: yaml.safe_load(p.read_text()) for p in sorted((root / "examples").glob("*.yaml"))}
snap = yaml.safe_load((root / "tests/expected_refusals.yaml").read_text())
bad = 0

if set(snap) != set(fws):
    print("FAIL framework set differs from snapshot: %s" % sorted(set(snap) ^ set(fws))); bad += 1
for name, fw in fws.items():
    derived = sorted(g for g, d in goldens.items() if F.expected(fw, d) == "refused")
    if derived != sorted(snap.get(name, [])):
        print("FAIL %s: derived refusals %s, snapshot %s" % (name, derived, sorted(snap.get(name, [])))); bad += 1

# both directions: take a capability away (match -> refused) and add one (refused -> match)
gpy, gpml = copy.deepcopy(fws["gpy"]), copy.deepcopy(fws["gpml"])
del gpy["_caps"]["kernels"]["white"]
if F.expected(gpy, goldens["ftc_rbf_white"]) != "refused":
    print("FAIL removing white did not flip ftc_rbf_white to refused"); bad += 1
gpml["_caps"]["approximations"]["dtcvar"] = None
gpml["_caps"]["kernels"]["rbf"] = None
if F.expected(gpml, goldens["dtcvar_mean_constant"]) != "match":
    print("FAIL adding dtcvar did not flip dtcvar_mean_constant to match"); bad += 1
# options: ARD on rbf is exercised by dtcvar_ard_data; claiming no options must refuse it
gpy["_caps"]["approximations"]["dtcvar"] = None
gpy["_caps"]["kernels"].update(rbf=set(), add=None, bias=None)
if F.expected(gpy, goldens["dtcvar_ard_data"]) != "refused":
    print("FAIL rbf without the ard option did not refuse an ARD golden"); bad += 1

# validation: unknown names and unknown options are rejected
import tempfile
for text, what in (("kernels: [nosuchkernel]", "unknown kernel"), ("kernels: [{rbf: {options: [warp]}}]", "unknown option")):
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "x.yaml").write_text("name: x\ncapabilities:\n  " + text + "\n")
        try:
            F.load_frameworks(pathlib.Path(d)); print("FAIL %s was accepted" % what); bad += 1
        except ValueError:
            pass

print("framework files: %d frameworks, %d goldens, %s" % (len(fws), len(goldens), "FAIL" if bad else "ok"))
sys.exit(1 if bad else 0)
