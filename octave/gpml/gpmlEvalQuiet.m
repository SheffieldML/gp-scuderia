function gpmlEvalQuiet(wire)
% One end-to-end GPML evaluation with output suppressed (the unit that is timed).
evalc('gpmlReference(wire);');
end
