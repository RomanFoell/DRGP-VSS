function [X_dyn_exogen,X_dyn_exogen_ref,y_scaled,y_ref_scaled,scaleshift,D,D_hidden,D_output] = deep_ssarx(X,X_ref,y,y_ref,order,non_rec)

% deep state space NARX structure
X_dyn_exogen_merged = [];
MU_X_dyn = [];
SIGMA_X_dyn = [];
nX = size(X,2);
[X_scaled,MU_X,SIGMA_X] = zscore(X');
X_scaled = X_scaled';
X_ref_scaled = do_scaleshift(X_ref,MU_X,SIGMA_X);
for i = 1:order
    MU_X_dyn = [MU_X_dyn,MU_X];
    SIGMA_X_dyn = [SIGMA_X_dyn,SIGMA_X];
    X_dyn_exogen_merged = [X_dyn_exogen_merged;[X_scaled(:,order + 1 - i:end),X_ref_scaled(:,1:end - i + 1)]];
end%for
X_dyn_exogen = X_dyn_exogen_merged(:,1:nX - order);
X_dyn_exogen_ref = X_dyn_exogen_merged(:,nX - order + 2:end);

if strcmp(non_rec,'on') 
    D = size(X_dyn_exogen,1);
else
    D = order + size(X_dyn_exogen,1);
end%if
D_hidden = 2 * order;
D_output = order;

[y_scaled,MU_y,SIGMA_y] = zscore(y);
y_ref_scaled = do_scaleshift(y_ref',MU_y,SIGMA_y)';

scaleshift.MU_X = MU_X;
scaleshift.SIGMA_X = SIGMA_X;
scaleshift.MU_y = MU_y;
scaleshift.SIGMA_y = SIGMA_y;

end