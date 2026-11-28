function doc = gpscuderiaKernelForm(doc)

% GPSCUDERIAKERNELFORM Rewrite mean.linear_basis as bias and linear kernels.
% FORMAT
% DESC mirrors python/gp_scuderia/kernel_form.py (CIP-0009). A Gaussian prior
% N(m0, S0) on the basis coefficients is the same GP as the mean F m0 plus the
% kernel F S0 F', so a document is rewritten exactly when that kernel is a sum of
% bias and linear terms: every column is the intercept or one linear term, the
% linear columns cover every input with equal variance, S0 is diagonal and m0 is
% zero except on the intercept. A fixed beta is realised only for an
% intercept-only basis, as mean.constant. Everything else raises
% gpscuderia:unsupported; nothing nearby is substituted.
% ARG doc : document from gpscuderiaRead with a mean.linear_basis mean.
% RETURN doc : equivalent document with a constant or no mean.
%
% SEEALSO : gpscuderiaToModel
%
% COPYRIGHT : Neil D. Lawrence, 2026

node = doc.mean; d = doc.input_dim;
powers = reshape(node.basis.powers, [], d);
mode = node.beta.mode;
doc = rmfield(doc, 'mean');
if strcmp(mode, 'fixed')
  if ~isequal(powers, zeros(1, d))
    error('gpscuderia:unsupported', ['mean.linear_basis fixed beta is realised only for an ' ...
          'intercept-only basis (a constant mean)']);
  end
  doc.mean = struct('type', 'mean.constant', 'value', node.beta.value(1));
  return
end
if ~strcmp(mode, 'gaussian_prior')
  error('gpscuderia:unsupported', ['mean.linear_basis beta.mode %s needs the coefficients ' ...
        'estimated, which this framework cannot do'], mode);
end
nb = size(powers, 1);
m0 = reshape(node.beta.prior_mean, nb, 1);
S0 = reshape(node.beta.prior_cov, nb, nb);
if any(any(S0 - diag(diag(S0)) ~= 0))
  error('gpscuderia:unsupported', 'mean.linear_basis prior_cov must be diagonal (no correlated coefficients)');
end
icol = 0; lin = zeros(1, d);
for j = 1:nb
  row = powers(j, :);
  if ~any(row)
    icol = j;
  elseif sum(row) == 1 && max(row) == 1
    lin(find(row)) = j; %#ok<FNDSB>
  else
    error('gpscuderia:unsupported', ['mean.linear_basis column [%s] is not an intercept or a linear ' ...
          'term, which bias and linear kernels cannot express'], num2str(row));
  end
end
extra = {};
if icol > 0
  extra{end+1} = struct('type', 'kern.bias', 'variance', S0(icol, icol));
end
if any(lin)
  if ~all(lin)
    error('gpscuderia:unsupported', ['mean.linear_basis linear columns must cover every input ' ...
          '(kern.linear spans all inputs)']);
  end
  vs = unique(diag(S0)(lin));
  if numel(vs) ~= 1
    error('gpscuderia:unsupported', ['mean.linear_basis linear columns need one prior variance ' ...
          '(kern.linear has one variance)']);
  end
  if any(m0(lin) ~= 0)
    error('gpscuderia:unsupported', ['mean.linear_basis prior_mean must be zero on linear columns ' ...
          '(the mean can only be constant)']);
  end
  extra{end+1} = struct('type', 'kern.linear', 'variance', vs);
end
if icol > 0 && m0(icol) ~= 0
  doc.mean = struct('type', 'mean.constant', 'value', m0(icol));
end
k = doc.kernel;
if strcmp(k.type, 'kern.add'), base = k.parts; else base = {k}; end
doc.kernel = struct('type', 'kern.add', 'parts', {[base(:)', extra]});
