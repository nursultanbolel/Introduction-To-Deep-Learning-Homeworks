function [Output] = sigmDerivative(X)
Output = 1 ./ (1 + exp(-X));
Output = Output.*(1-Output);
end

