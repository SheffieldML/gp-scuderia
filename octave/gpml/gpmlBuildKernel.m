function [cov, hyp, dim] = gpmlBuildKernel(k)
% Translate a wire kernel into a GPML covariance spec and its log-hyperparameters.
% Anything GPML cannot realise raises gpscuderia:unsupported.
switch k.type
  case 'kern.rbf'
    if isfield(k, 'ARD') && k.ARD
      ls = k.lengthscale(:)';
      cov = {@covSEard}; hyp = [log(ls), log(sqrt(k.variance))];
    else
      cov = {@covSEiso}; hyp = [log(k.lengthscale), log(sqrt(k.variance))];
    end
  case {'kern.matern32', 'kern.matern52'}
    if isfield(k, 'ARD') && k.ARD, error('gpscuderia:unsupported', '%s with ARD is not realised by GPML covMaterniso', k.type); end
    d = 3; if strcmp(k.type, 'kern.matern52'), d = 5; end
    cov = {@covMaterniso, d}; hyp = [log(k.lengthscale), log(sqrt(k.variance))];
  case 'kern.bias'
    cov = {@covConst}; hyp = log(sqrt(k.variance));
  case 'kern.white'
    cov = {@covNoise}; hyp = log(sqrt(k.variance));
  case {'kern.add', 'kern.prod'}
    parts = k.parts; if isstruct(parts), parts = num2cell(parts); end
    cs = cell(1, numel(parts)); hyp = [];
    for i = 1:numel(parts)
      [c, h] = gpmlBuildKernel(parts{i}); cs{i} = c; hyp = [hyp, h(:)'];
    end
    if strcmp(k.type, 'kern.add'), cov = {@covSum, cs}; else cov = {@covProd, cs}; end
  otherwise
    error('gpscuderia:unsupported', '%s is not realised by GPML', k.type);
end
end
