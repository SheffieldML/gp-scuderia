"""GPy adapter: wire document <-> GPy model, plus reference computation.

GPy is the independent oracle for the numerical fixtures: wire parameters are
GPy-natural (positive, untransformed), so this adapter does no unit conversion.
"""
import numpy as np
import GPy

_KERN = {
    "kern.rbf": GPy.kern.RBF,
    "kern.matern32": GPy.kern.Matern32,
    "kern.matern52": GPy.kern.Matern52,
    "kern.white": GPy.kern.White,
    "kern.bias": GPy.kern.Bias,
    "kern.linear": GPy.kern.Linear,
}
_TAG = {v: k for k, v in _KERN.items()}
# GPy sparse regression selects the approximation by inference method.
_INFER = {"fitc": "FITC", "dtcvar": None}


def _kern_from_wire(node, input_dim):
    t = node["type"]
    if t in ("kern.add", "kern.prod"):
        parts = [_kern_from_wire(p, input_dim) for p in node["parts"]]
        out = parts[0]
        for p in parts[1:]:
            out = out + p if t == "kern.add" else out * p
        return out
    cls = _KERN[t]
    kw = {"variance": node["variance"]}
    if t in ("kern.rbf", "kern.matern32", "kern.matern52"):
        ls = node["lengthscale"]
        ard = bool(node.get("ARD", isinstance(ls, list)))
        kw.update(lengthscale=ls, ARD=ard)
    if "active_dims" in node:
        kw["active_dims"] = node["active_dims"]
    return cls(input_dim, **kw)


def from_wire(doc):
    d = doc["input_dim"]
    kern = _kern_from_wire(doc["kernel"], d)
    X = np.array(doc["data"]["X"], float)
    Y = np.array(doc["data"]["Y"], float)
    noise = doc["likelihood"]["variance"]
    approx = doc.get("approximation", "ftc")
    if approx == "ftc":
        return GPy.models.GPRegression(X, Y, kern, noise_var=noise)
    Z = np.array(doc["inducing_inputs"], float)
    if approx == "dtcvar":
        m = GPy.models.SparseGPRegression(X, Y, kern, Z=Z)
    elif approx == "fitc":
        m = GPy.models.SparseGPRegression(X, Y, kern, Z=Z)
        m.inference_method = GPy.inference.latent_function_inference.FITC()
    else:
        raise NotImplementedError("GPy adapter: approximation %s" % approx)
    m.Gaussian_noise.variance = noise
    m.Z.fix()
    return m


def reference(doc, Xnew):
    """Compute log-likelihood and predictive mean/variance for doc at Xnew."""
    m = from_wire(doc)
    Xnew = np.array(Xnew, float)
    mean, var = m.predict(Xnew)
    return {
        "log_likelihood": float(m.log_likelihood()),
        "Xnew": Xnew.tolist(),
        "mean": mean.ravel().tolist(),
        "var": var.ravel().tolist(),
    }
