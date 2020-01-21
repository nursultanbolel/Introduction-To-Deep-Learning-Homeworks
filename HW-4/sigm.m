function [O] = sigm(X)
O = 1 ./ (1 + exp(-X));
end

