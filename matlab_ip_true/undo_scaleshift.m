function [X_out] = undo_scaleshift(X,MU_X,SIGMA_X)

nX = size(X,2);
X_out = X .* repmat(SIGMA_X',1,nX) + repmat(MU_X',1,nX);

end