"""GPc entry point for the comparison harness (needs the gpc Python package)."""


def reference(doc, Xnew):
    from gpc import interchange as I
    from ._common import UnsupportedByFramework
    from .kernel_form import kernel_form, NotKernelForm
    if doc.get("mean", {}).get("type") == "mean.linear_basis":
        # GPc has a constant mean and bias and linear kernels only. A realisable Gaussian
        # prior on the coefficients is rewritten as an exact kernel; anything else raises
        # UnsupportedByFramework.
        try:
            doc = kernel_form(doc)
        except NotKernelForm as e:
            raise UnsupportedByFramework(str(e))
        def has_linear(k):
            return k["type"] == "kern.linear" or any(has_linear(q) for q in k.get("parts", []))
        if has_linear(doc["kernel"]):
            raise UnsupportedByFramework("the GPc interchange reader has no kern.linear yet")
    return I.reference(I.from_wire(doc), Xnew)
