"""GPc entry point for the comparison harness (needs the gpc Python package)."""


def reference(doc, Xnew):
    from gpc import interchange as I
    return I.reference(I.from_wire(doc), Xnew)
