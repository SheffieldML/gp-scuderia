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
    if t == "kern.coregionalize":
        W = np.array(node["W"], float)
        return GPy.kern.Coregionalize(1, output_dim=node["output_dim"], rank=W.shape[1],
                                      W=W, kappa=np.array(node["kappa"], float),
                                      active_dims=node["active_dims"])
    cls = _KERN[t]
    dims = node.get("active_dims")
    if dims is not None:
        input_dim = len(dims)
    kw = {"variance": node["variance"]}
    if t in ("kern.rbf", "kern.matern32", "kern.matern52"):
        ls = node["lengthscale"]
        ard = bool(node.get("ARD", isinstance(ls, list)))
        kw.update(lengthscale=ls, ARD=ard)
    if "active_dims" in node:
        kw["active_dims"] = node["active_dims"]
    return cls(input_dim, **kw)


class _FixedStandardize(GPy.util.normalizer.Standardize):
    """GPy Standardize with the wire's mean and std instead of the data's."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.array([[mean]], float)
        self.std = np.array([[std]], float)

    def scale_by(self, Y):
        pass


def _target_transform(doc):
    """(normalizer, mean_function) kwargs for a GPy model from the wire fields."""
    if "mean" in doc and "normalizer" in doc:
        raise NotImplementedError("mean and normalizer together are not in schema 0.1")
    norm, mf = None, None
    if "normalizer" in doc:
        n = doc["normalizer"]
        if n["type"] == "normalizer.center":
            norm = _FixedStandardize(n["mean"], 1.0)
        elif n["type"] == "normalizer.standardize":
            norm = _FixedStandardize(n["mean"], n["std"])
        else:
            raise NotImplementedError(n["type"])
    if "mean" in doc:
        if doc["mean"]["type"] != "mean.constant":
            raise NotImplementedError(doc["mean"]["type"])
        mf = GPy.mappings.Constant(doc["input_dim"], 1, value=doc["mean"]["value"])
    return norm, mf


def from_wire(doc):
    d = doc["input_dim"]
    norm, mf = _target_transform(doc)
    kern = _kern_from_wire(doc["kernel"], d)
    X = np.array(doc["data"]["X"], float)
    Y = np.array(doc["data"]["Y"], float)
    lik = doc["likelihood"]
    noise = lik["variance"]
    approx = doc.get("approximation", "ftc")
    if lik["type"] == "likelihoods.mixed_gaussian":
        if approx != "ftc":
            raise NotImplementedError("mixed_gaussian with %s" % approx)
        idx = X[:, lik["output_index_column"]].astype(int)[:, None]
        mixed = GPy.likelihoods.MixedNoise([GPy.likelihoods.Gaussian(variance=v) for v in noise])
        return GPy.core.GP(X, Y, kern, mixed, Y_metadata={"output_index": idx},
                           normalizer=norm, mean_function=mf)
    if approx == "ftc":
        return GPy.models.GPRegression(X, Y, kern, noise_var=noise, normalizer=norm,
                                       mean_function=mf)
    Z = np.array(doc["inducing_inputs"], float)
    if approx == "dtcvar":
        m = GPy.models.SparseGPRegression(X, Y, kern, Z=Z, normalizer=norm, mean_function=mf)
    elif approx == "fitc":
        m = GPy.models.SparseGPRegression(X, Y, kern, Z=Z, normalizer=norm, mean_function=mf)
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
    dims = [int(i) for i in k.active_dims]
    if isinstance(k, GPy.kern.Coregionalize):
        return {"type": "kern.coregionalize", "output_dim": int(k.output_dim),
                "W": np.asarray(k.W.values).tolist(),
                "kappa": np.asarray(k.kappa.values).tolist(), "active_dims": dims}
    tag = _TAG.get(type(k))
    if tag is None:
        raise WireExportError("unsupported GPy kernel %s" % type(k).__name__)
    node = {"type": tag, "variance": float(k.variance[0])}
    if dims != list(range(input_dim)):
        node["active_dims"] = dims
    if tag in ("kern.rbf", "kern.matern32", "kern.matern52"):
        node["lengthscale"] = [float(v) for v in k.lengthscale] if k.ARD else float(k.lengthscale[0])
        node["ARD"] = bool(k.ARD)
    return node


def _index_column(k):
    """Column carrying the output index: the coregionalize kernel's active dim."""
    cols = set()
    def walk(p):
        if isinstance(p, GPy.kern.Coregionalize):
            cols.add(int(p.active_dims[0]))
        for q in getattr(p, "parts", []):
            walk(q)
    walk(k)
    if len(cols) != 1:
        raise WireExportError("mixed noise needs one coregionalize index column")
    return cols.pop()


def to_wire(m):
    """Describe a GPy GPRegression or SparseGPRegression as a wire document.

    Parameters are read from the model. Anything the format cannot carry (a
    normalizer, a mean function, an unsupported kernel or inference method,
    several outputs, a non-Gaussian likelihood) raises WireExportError.
    """
    mixed = isinstance(m.likelihood, GPy.likelihoods.MixedNoise)
    if mixed and type(m) is GPy.core.GP:
        pass
    elif not isinstance(m, (GPy.models.GPRegression, GPy.models.SparseGPRegression)):
        raise WireExportError("unsupported GPy model %s" % type(m).__name__)
    norm = getattr(m, "normalizer", None)
    mf = getattr(m, "mean_function", None)
    if norm and mf is not None:
        raise WireExportError("mean and normalizer together are not in schema 0.1")
    if norm and not isinstance(norm, GPy.util.normalizer.Standardize):
        raise WireExportError("unsupported normalizer %s" % type(norm).__name__)
    if mf is not None and not isinstance(mf, GPy.mappings.Constant):
        raise WireExportError("unsupported mean function %s" % type(mf).__name__)
    if not mixed and not isinstance(m.likelihood, GPy.likelihoods.Gaussian):
        raise WireExportError("non-Gaussian likelihood")
    if m.Y.shape[1] != 1:
        raise WireExportError("only one output column is on the wire (stack outputs with an index column)")
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
    if mixed:
        gl = m.likelihood.likelihoods_list
        col = _index_column(m.kern)
        doc["likelihood"] = {"type": "likelihoods.mixed_gaussian",
                             "variance": [float(g.variance[0]) for g in gl],
                             "output_index_column": col}
    else:
        doc["likelihood"] = {"type": "likelihoods.gaussian",
                             "variance": float(m.likelihood.variance[0])}
    Y = np.asarray(m.Y)  # GPy keeps the original-scale Y; the normalized copy is Y_normalized
    if norm:
        doc["normalizer"] = {"type": "normalizer.standardize",
                             "mean": float(norm.mean.ravel()[0]), "std": float(norm.std.ravel()[0])}
    if mf is not None:
        doc["mean"] = {"type": "mean.constant", "value": float(np.asarray(mf.C.values).ravel()[0])}
    doc["data"] = {"X": np.asarray(m.X).tolist(), "Y": Y.tolist()}
    return doc


def reference(doc, Xnew):
    """Compute log-likelihood and predictive mean/variance for doc at Xnew."""
    m = from_wire(doc)
    Xnew = np.array(Xnew, float)
    meta = None
    lik = doc["likelihood"]
    if lik["type"] == "likelihoods.mixed_gaussian":
        meta = {"output_index": Xnew[:, lik["output_index_column"]].astype(int)[:, None]}
    mean, var = m.predict(Xnew, Y_metadata=meta)
    ll = float(m.log_likelihood())
    if doc.get("normalizer", {}).get("type") == "normalizer.standardize":
        # GPy scores the standardized targets; the wire reports log p(y).
        Y = np.asarray(doc["data"]["Y"], float)
        ll -= Y.size * float(np.log(doc["normalizer"]["std"]))
    return {
        "log_likelihood": ll,
        "Xnew": Xnew.tolist(),
        "mean": mean.ravel().tolist(),
        "var": var.ravel().tolist(),
    }
