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


class WireExportError(ValueError):
    """The GPy model cannot be described exactly by the wire format."""


def _kern_to_wire(k, input_dim):
    if isinstance(k, (GPy.kern.Add, GPy.kern.Prod)):
        tag = "kern.add" if isinstance(k, GPy.kern.Add) else "kern.prod"
        return {"type": tag, "parts": [_kern_to_wire(p, input_dim) for p in k.parts]}
    tag = _TAG.get(type(k))
    if tag is None:
        raise WireExportError("unsupported GPy kernel %s" % type(k).__name__)
    if list(k.active_dims) != list(range(input_dim)):
        raise WireExportError("active_dims %s are not on the wire" % list(k.active_dims))
    node = {"type": tag, "variance": float(k.variance[0])}
    if tag in ("kern.rbf", "kern.matern32", "kern.matern52"):
        node["lengthscale"] = [float(v) for v in k.lengthscale] if k.ARD else float(k.lengthscale[0])
        node["ARD"] = bool(k.ARD)
    return node


def to_wire(m):
    """Describe a GPy GPRegression or SparseGPRegression as a wire document.

    Parameters are read from the model. Anything the format cannot carry (a
    normalizer, a mean function, an unsupported kernel or inference method,
    several outputs, a non-Gaussian likelihood) raises WireExportError.
    """
    if not isinstance(m, (GPy.models.GPRegression, GPy.models.SparseGPRegression)):
        raise WireExportError("unsupported GPy model %s" % type(m).__name__)
    if getattr(m, "normalizer", None):
        raise WireExportError("normalizer is not yet on the wire (CIP-0001 item 3)")
    if getattr(m, "mean_function", None) is not None:
        raise WireExportError("mean functions are not yet on the wire (CIP-0001 item 3)")
    if not isinstance(m.likelihood, GPy.likelihoods.Gaussian):
        raise WireExportError("non-Gaussian likelihood")
    if m.Y.shape[1] != 1:
        raise WireExportError("multi-output models are not yet on the wire")
    d = m.input_dim
    doc = {"schema_version": "0.1", "input_dim": d, "output_dim": 1}
    if isinstance(m, GPy.models.SparseGPRegression):
        im = m.inference_method
        if isinstance(im, GPy.inference.latent_function_inference.FITC):
            approx = "fitc"
        elif isinstance(im, GPy.inference.latent_function_inference.VarDTC):
            approx = "dtcvar"
        else:
            raise WireExportError("unsupported inference %s" % type(im).__name__)
        doc.update(type="models.sparse_gp_regression", approximation=approx,
                   inducing_inputs=np.asarray(m.Z.values).tolist())
    else:
        doc.update(type="models.gp_regression", approximation="ftc")
    doc["kernel"] = _kern_to_wire(m.kern, d)
    doc["likelihood"] = {"type": "likelihoods.gaussian",
                         "variance": float(m.likelihood.variance[0])}
    doc["data"] = {"X": np.asarray(m.X).tolist(), "Y": np.asarray(m.Y).tolist()}
    return doc


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
