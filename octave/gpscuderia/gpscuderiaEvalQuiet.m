function gpscuderiaEvalQuiet(doc)
% One end-to-end GPmat evaluation (build model from the wire document, log likelihood, prediction) with output suppressed.
Xn = doc.x_reference.Xnew;
evalc('m = gpscuderiaToModel(doc); g = gpscuderiaReference(m, Xn);');
end
