function doc = gpscuderiaFromModel(model, template)

% GPSCUDERIAFROMMODEL Describe a GPmat model as an interchange document.
% FORMAT
% DESC reads parameters back from the model (so the round trip is checked,
% not copied). The kernel tree, data and inducing inputs come from template
% because GPmat does not retain the wire structure.
% ARG model : GPmat GP model.
% ARG template : document supplying structure (from gpscuderiaRead).
% RETURN doc : document with parameters read from model.
%
% SEEALSO : gpscuderiaToModel
%
% COPYRIGHT : Neil D. Lawrence, 2026

doc = template;
if isfield(doc, 'x_reference'), doc = rmfield(doc, 'x_reference'); end
leaves = kernLeaves(template.kernel);
vals = cell(1, numel(leaves));
comps = kernCompStructs(model.kern);
if numel(comps) ~= numel(leaves)
  error('gpscuderia:shape', 'model kernel has %d leaves, template has %d', numel(comps), numel(leaves));
end
for i = 1:numel(leaves)
  comp = comps{i};
  leaf = leaves{i};
  leaf.variance = comp.variance;
  if isfield(comp, 'inputScales')
    leaf.lengthscale = 1 ./ sqrt(comp.inputScales(:)');
  elseif isfield(comp, 'inverseWidth')
    leaf.lengthscale = 1/sqrt(comp.inverseWidth);
  elseif isfield(comp, 'lengthScale')
    leaf.lengthscale = comp.lengthScale;
  end
  vals{i} = leaf;
end
doc.kernel = putLeaves(template.kernel, vals, 1);
if isfield(model, 'beta') && ~isempty(model.beta)
  doc.likelihood.variance = 1/model.beta;
end


function comps = kernCompStructs(kern)
% Leaf kernel structures of a (possibly nested) GPmat kernel, in document order.
if isfield(kern, 'comp')
  comps = {};
  for i = 1:numel(kern.comp)
    comps = [comps, kernCompStructs(kern.comp{i})]; %#ok<AGROW>
  end
else
  comps = {kern};
end


function leaves = kernLeaves(node)
if isfield(node, 'parts')
  leaves = {};
  for i = 1:numel(node.parts)
    leaves = [leaves, kernLeaves(node.parts{i})]; %#ok<AGROW>
  end
else
  leaves = {node};
end


function [node, next] = putLeaves(node, vals, next)
if isfield(node, 'parts')
  for i = 1:numel(node.parts)
    [node.parts{i}, next] = putLeaves(node.parts{i}, vals, next);
  end
else
  node = vals{next};
  next = next + 1;
end
