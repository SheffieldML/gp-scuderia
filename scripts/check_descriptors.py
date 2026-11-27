"""Check the task descriptor function against hand-computed values (CIP-0005).

Five points on a line, X = 0..4, RBF with variance 2 and lengthscale 2, noise 0.5. By hand:
signal-to-noise is 2 / 0.5 = 4; every nearest-neighbour distance is 1, so lengthscale over
spacing is 2; K + 0.5 I is 5 x 5 with entries 2 exp(-(i-j)^2 / 8) on the off-diagonals. The
condition number is computed here from numpy's SVD, a different route from the eigvalsh in the
function under test, and checked to 1e-9.
"""
import pathlib, sys
import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(root / "python"))
from benchmarks import descriptors, COND_MAX_N

bad = 0
def check(label, ok):
    global bad
    print(("ok   " if ok else "FAIL ") + label); bad += not ok

X = np.arange(5.0)[:, None]
doc = {"data": {"X": X.tolist()}, "kernel": {"type": "kern.rbf", "variance": 2.0, "lengthscale": 2.0},
       "likelihood": {"variance": 0.5}}
d = descriptors(doc)
i = np.arange(5.0)
K = 2.0 * np.exp(-((i[:, None] - i[None, :]) ** 2) / 8.0) + 0.5 * np.eye(5)
check("n and input_dim", d["n"] == 5 and d["input_dim"] == 1)
check("signal to noise is 4", d["signal_to_noise"] == 4.0)
check("lengthscale over spacing is 2", abs(d["lengthscale_over_spacing"] - 2.0) < 1e-12)
check("log10 condition matches numpy SVD", abs(d["log10_condition"] - np.log10(np.linalg.cond(K))) < 1e-9)

arddoc = {"data": {"X": [[0, 0], [1, 0], [0, 10]]}, "kernel": {"type": "kern.rbf", "variance": 1.0, "lengthscale": [1.0, 10.0], "ARD": True},
          "likelihood": {"variance": 1.0}}
# scaled points: (0,0), (1,0), (0,1): nearest-neighbour distances 1, 1, 1 -> ratio 1
check("ARD lengthscales respected", abs(descriptors(arddoc)["lengthscale_over_spacing"] - 1.0) < 1e-12)

dup = {"data": {"X": [[0.0], [0.0], [1.0]]}, "kernel": {"type": "kern.rbf", "variance": 1.0, "lengthscale": 1.0},
       "likelihood": {"variance": 1.0}}
check("duplicated inputs do not give an infinite ratio", np.isfinite(descriptors(dup)["lengthscale_over_spacing"]))
comp = {"data": {"X": [[0.0], [1.0]]}, "kernel": {"type": "kern.sum", "parts": [{"type": "kern.rbf", "variance": 1.0, "lengthscale": 1.0}]},
        "likelihood": {"variance": 1.0}}
dc = descriptors(comp)
check("compound kernels give null kernel-dependent descriptors", dc["log10_condition"] is None and dc["signal_to_noise"] is None)
sys.exit(1 if bad else 0)
