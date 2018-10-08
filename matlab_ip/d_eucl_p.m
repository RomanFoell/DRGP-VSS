function [deff] = d_eucl_p(X)

% euclidean distance squared

nX = size(X,2);
% less stable, but more fast
temp = repmat(X,[1,1,nX]);
deff = squeeze(sum(temp - permute(temp,[1 3 2]),1));

end