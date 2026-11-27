"""GPyTorch adapter: wire document -> exact GPyTorch model, plus reference computation.

GPyTorch kernels have no variance of their own, so every leaf is wrapped in a
ScaleKernel (outputscale = wire variance). There is no white kernel (noise lives in
the likelihood), no exact FITC, and no collapsed variational bound, so kern.white,
sparse approximations and coregionalisation raise UnsupportedByFramework rather
than being folded into something that differs numerically.
"""
import numpy as np
import torch
import gpytorch
from ._common import (UnsupportedByFramework, check_single_output_ftc_or,
                      reference_from_transformed, transformed_Y)

torch.set_default_dtype(torch.float64)


def _kern(node, d):
    t = node["type"]
    if t in ("kern.add", "kern.prod"):
        parts = [_kern(p, d) for p in node["parts"]]
        return (gpytorch.kernels.AdditiveKernel if t == "kern.add"
                else gpytorch.kernels.ProductKernel)(*parts)
    if t == "kern.white":
        raise UnsupportedByFramework(
            "GPyTorch has no white kernel; its noise is the likelihood, so kern.white would "
            "change the model if folded into the likelihood variance")
    if t == "kern.coregionalize":
        raise UnsupportedByFramework("kern.coregionalize: GPyTorch multitask models use a Kronecker structure")
    dims = node.get("active_dims")
    kw = {"active_dims": torch.tensor(dims)} if dims is not None else {}
    nd = len(dims) if dims is not None else d
    if t in ("kern.rbf", "kern.matern32", "kern.matern52"):
        ls = node["lengthscale"]
        ard = nd if isinstance(ls, list) else None
        if t == "kern.rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=ard, **kw)
        else:
            base = gpytorch.kernels.MaternKernel(nu=1.5 if t == "kern.matern32" else 2.5,
                                                 ard_num_dims=ard, **kw)
        base.lengthscale = torch.tensor(ls if isinstance(ls, list) else [[float(ls)]]).reshape(1, -1)
        k = gpytorch.kernels.ScaleKernel(base)
        k.outputscale = node["variance"]
        return k
    if t == "kern.bias":
        k = gpytorch.kernels.ConstantKernel(**kw)
        k.constant = torch.tensor(float(node["variance"]))
        return k
    if t == "kern.linear":
        k = gpytorch.kernels.LinearKernel(**kw)
        k.variance = torch.tensor(float(node["variance"]))
        return k
    raise UnsupportedByFramework("GPyTorch adapter: %s" % t)


class _Exact(gpytorch.models.ExactGP):
    def __init__(self, X, y, lik, kern):
        super().__init__(X, y, lik)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kern

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def from_wire(doc):
    check_single_output_ftc_or(doc, ("ftc",))
    kern = _kern(doc["kernel"], doc["input_dim"])
    X = torch.tensor(np.array(doc["data"]["X"], float))
    y = torch.tensor(transformed_Y(doc).ravel())
    lik = gpytorch.likelihoods.GaussianLikelihood()
    lik.noise = doc["likelihood"]["variance"]
    return _Exact(X, y, lik, kern), lik


def reference(doc, Xnew):
    model, lik = from_wire(doc)
    Xn = np.array(Xnew, float)
    model.train(); lik.train()
    # max_cholesky_size: above 800 points GPyTorch switches to conjugate gradients and stochastic
    # log-determinants, which is an approximation; a conformance reference must be the exact value.
    with gpytorch.settings.cholesky_jitter(1e-12, 1e-12), gpytorch.settings.max_cholesky_size(10 ** 9):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        ll = mll(model(*model.train_inputs), model.train_targets).item() * len(model.train_targets)
        model.eval(); lik.eval()
        with torch.no_grad():
            pred = lik(model(torch.tensor(Xn)))
    return reference_from_transformed(doc, ll, pred.mean.numpy(), pred.variance.numpy(), Xn)


def _modules(k, node, path, out):
    """wire path -> (module, property name) for each differentiable parameter."""
    if "parts" in node:
        for i, (kp, n) in enumerate(zip(k.kernels, node["parts"])):
            _modules(kp, n, "%s.parts.%d" % (path, i), out)
        return
    t = node["type"]
    if t in ("kern.rbf", "kern.matern32", "kern.matern52"):
        out[path + ".variance"] = (k, "outputscale")
        out[path + ".lengthscale"] = (k.base_kernel, "lengthscale")
    elif t == "kern.bias":
        out[path + ".variance"] = (k, "constant")
    elif t == "kern.linear":
        out[path + ".variance"] = (k, "variance")


def gradient(doc):
    """dLL/d(wire parameter): autograd to the raw (unconstrained) parameter, then divide by
    d(natural value)/d(raw), taken from GPyTorch's own constraint (elementwise)."""
    from .params import wire_paths
    wire_paths(doc)
    model, lik = from_wire(doc)
    mods = {}
    _modules(model.covar_module, doc["kernel"], "kernel", mods)
    mods["likelihood.variance"] = (lik, "noise")
    model.train(); lik.train()
    with gpytorch.settings.cholesky_jitter(1e-12, 1e-12), gpytorch.settings.max_cholesky_size(10 ** 9):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        ll = mll(model(*model.train_inputs), model.train_targets) * len(model.train_targets)
    out = {}
    for path, (mod, prop) in mods.items():
        raw = getattr(mod, "raw_" + prop)
        dll = torch.autograd.grad(ll, raw, retain_graph=True)[0]
        dnat = torch.autograd.grad(getattr(mod, prop).sum(), raw)[0]
        g = (dll / dnat).detach().numpy().ravel()
        out[path] = g.tolist() if g.size > 1 else float(g[0])
    return out
