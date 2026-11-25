function n2 = dist2(x, c)
% DIST2 Squared Euclidean distances between rows of x and rows of c.
% Minimal stand-in for Netlab's dist2 so GPmat runs under Octave in CI
% without installing Netlab. Same semantics: n2(i,j) = ||x_i - c_j||^2.
[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
  error('Data dimension does not match dimension of centres')
end
n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
     ones(ndata, 1) * sum((c.^2)', 1) - 2 .* (x * (c'));
n2(n2 < 0) = 0;
