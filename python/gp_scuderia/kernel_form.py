"""Pure-Python rewrite of mean.linear_basis into bias and linear kernels (CIP-0009).

No dependencies, so the capability checks can use it without numpy.
"""


class NotKernelForm(ValueError):
    """The document has no exact bias/linear kernel form."""


def kernel_form(doc):
    """Rewrite a mean.linear_basis document as an equivalent bias/linear kernel document.

    For frameworks with only a constant mean and bias and linear kernels (GPc, GPmat). A
    Gaussian prior N(m0, S0) on beta is the same GP as the mean F m0 plus the kernel
    F S0 F^T, so the rewrite is exact when that kernel is a sum of bias and linear terms:

    * every basis column is the intercept or a single linear term x_i, and the linear
      columns cover all input dimensions (one kern.linear spans every input);
    * S0 is diagonal, with equal variances on the linear columns (kern.linear has one
      variance);
    * m0 is zero except on the intercept (the only mean these frameworks have is constant).

    A fixed beta is realised only for an intercept-only basis, as mean.constant. Estimated
    modes need beta itself and are refused, as is anything else: nothing nearby is substituted.
    Returns a new document without mean.linear_basis; raises NotKernelForm (adapters turn it into UnsupportedByFramework) otherwise.
    """
    node, d = doc["mean"], doc["input_dim"]
    beta, powers = node["beta"], [list(map(int, r)) for r in node["basis"]["powers"]]
    mode = beta["mode"]
    out = {k: v for k, v in doc.items() if k != "mean"}
    intercept = [0] * d
    if mode == "fixed":
        if powers != [intercept]:
            raise NotKernelForm(
                "mean.linear_basis fixed beta is realised only for an intercept-only basis (a constant mean)")
        out["mean"] = {"type": "mean.constant", "value": float(beta["value"][0])}
        return out
    if mode != "gaussian_prior":
        raise NotKernelForm(
            "mean.linear_basis beta.mode %s needs the coefficients estimated, which this framework cannot do" % mode)
    m0 = [float(v) for v in beta["prior_mean"]]
    S0 = [[float(v) for v in row] for row in beta["prior_cov"]]
    if any(S0[i][j] != 0 for i in range(len(S0)) for j in range(len(S0)) if i != j):
        raise NotKernelForm("mean.linear_basis prior_cov must be diagonal (no correlated coefficients)")
    icol, lin = None, {}
    for j, row in enumerate(powers):
        if row == intercept:
            icol = j
        elif sum(row) == 1 and max(row) == 1:
            lin[row.index(1)] = j
        else:
            raise NotKernelForm(
                "mean.linear_basis column %s is not an intercept or a linear term, which bias and linear kernels cannot express" % row)
    parts = []
    if icol is not None:
        parts.append({"type": "kern.bias", "variance": float(S0[icol][icol])})
    if lin:
        if sorted(lin) != list(range(d)):
            raise NotKernelForm("mean.linear_basis linear columns must cover every input (kern.linear spans all inputs)")
        vs = {float(S0[j][j]) for j in lin.values()}
        if len(vs) != 1:
            raise NotKernelForm("mean.linear_basis linear columns need one prior variance (kern.linear has one variance)")
        if any(m0[j] != 0 for j in lin.values()):
            raise NotKernelForm("mean.linear_basis prior_mean must be zero on linear columns (the mean can only be constant)")
        parts.append({"type": "kern.linear", "variance": vs.pop()})
    if icol is not None and m0[icol] != 0:
        out["mean"] = {"type": "mean.constant", "value": float(m0[icol])}
    k = doc["kernel"]
    base = list(k["parts"]) if k["type"] == "kern.add" else [k]
    out["kernel"] = {"type": "kern.add", "parts": base + parts}
    return out
