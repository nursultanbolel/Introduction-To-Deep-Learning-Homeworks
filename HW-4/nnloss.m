function y = nnloss(x, t, dzdy)
% VL_NNEUCLIDEANLOSS computes the L2 Loss
%  Y = VL_NNEUCLIDEANLOSS(X, T) computes the Euclidean Loss
%  (also known as the L2 loss) between an N x 1 array of input
%  predictions, X and an N x 1 array of targets, T. The output
%  Y is a scalar value.
%
% Copyright (C) 2017 Samuel Albanie
% All rights reserved.

instanceWeights = ones(size(x)) ;

% residuals
res = x - t ;


% %huber loss loss
% sigma = 1 ;
% absDelta = abs(res) ;
% sigma2 = sigma ^ 2 ;
% linearRegion = (absDelta > 1. / sigma2) ;
% 
% if  dzdy==0
%     absDelta(linearRegion) = absDelta(linearRegion) - 0.5 / sigma2 ;
%     absDelta(~linearRegion) = 0.5 * sigma2 * absDelta(~linearRegion) .^ 2 ;
%     y = instanceWeights(:)' * absDelta(:) ;
% else
%     res(linearRegion) = sign(res(linearRegion));
%     res(~linearRegion) = sigma2 * res(~linearRegion) ;
%     y = res;
% end
% 

%euclidean loss
if dzdy==0
  y = (1/2) * instanceWeights(:)' * res(:).^2 ;
else
  y = res  ;
end

end

