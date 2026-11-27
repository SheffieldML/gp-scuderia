function t = timeEval(f, spec, cold)
% Wall-clock timing of the evaluation handle f: warmup untimed calls, then spec.repeats timed calls.
% Returns the timing struct of the result contract (seconds_cold is supplied by the caller).
if isfield(spec, 'warmup'), nw = spec.warmup; else nw = 1; end
for k = 1:nw, f(); end
ts = zeros(1, spec.repeats);
for k = 1:spec.repeats
  t0 = tic; f(); ts(k) = toc(t0);
end
s = sort(ts); n = numel(s);
q = @(p) interp1(((1:n) - 0.5) / n, s, p, 'linear', 'extrap');
if n == 1, qs = [s s s]; else qs = [q(0.25) q(0.5) q(0.75)]; end
t = struct('seconds_cold', cold, 'seconds_median', median(ts), 'seconds_iqr', qs(3) - qs(1), 'repeats', spec.repeats);
end
