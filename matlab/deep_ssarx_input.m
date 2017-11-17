function [X_inputs,SIGMA_inputs] = deep_ssarx_input(X_exogen,hyp,layers,order,non_rec)

% deep state space ARX structure

X_inputs = cell(layers - 1,1);
SIGMA_inputs = cell(layers - 1,1);
X_dyn = X_exogen;
SIGMA = log(exp(sqrt(zeros(size(X_exogen)))) - 1);
for i = 1:order
    X_dyn = [X_dyn;hyp.layer1.mu(order + 1 - i:end - i)];
    SIGMA = [SIGMA;hyp.layer1.sigma(order + 1 - i:end - i)];
end%for

X_inputs{1} = X_dyn;
SIGMA_inputs{1} = SIGMA;

for j = 2:layers
    X_inputs{j} =[hyp.(strcat('layer',num2str(j - 1))).mu(order + 1:end);X_inputs{j - 1}(end-order+1:end-1,:)];
    SIGMA_inputs{j} = [hyp.(strcat('layer',num2str(j - 1))).sigma(order + 1:end);SIGMA_inputs{j - 1}(end-order+1:end-1,:)];
    for i = 1:order
        X_inputs{j} = [X_inputs{j};hyp.(strcat('layer',num2str(j))).mu(order + 1 - i:end - i)];
        SIGMA_inputs{j} = [SIGMA_inputs{j};hyp.(strcat('layer',num2str(j))).sigma(order + 1 - i:end - i)];
    end%for
end%for

X_inputs{layers + 1} =[hyp.(strcat('layer',num2str(layers))).mu(order + 1:end);X_inputs{layers}(end-order+1:end-1,:)];
SIGMA_inputs{layers + 1} = [hyp.(strcat('layer',num2str(layers))).sigma(order + 1:end);SIGMA_inputs{layers}(end-order+1:end-1,:)];
 
if strcmp(non_rec,'on') 
    X_inputs{1} = X_exogen;
    SIGMA_inputs{1} = log(exp(sqrt(zeros(size(X_exogen)))) - 1);
else
    %
end%if