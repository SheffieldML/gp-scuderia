"""GPc entry point for the comparison harness (needs the gpc Python package)."""
from ._common import UnsupportedByFramework


def reference(doc, Xnew):
    from gpc import interchange as I
    return I.reference(I.from_wire(doc), Xnew)


def _latent_op(name, doc):
    from gpc import interchange as I
    fn = getattr(I, name, None)
    if fn is None:
        raise UnsupportedByFramework("the installed gpc.interchange has no %s operation" % name)
    return fn(I.from_wire(doc))


def bound(doc):
    """CBayesGplvm: varDtcBound() is the bound without KL; the KL is reported separately."""
    return _latent_op("bound", doc)


def psi(doc):
    return _latent_op("psi", doc)
