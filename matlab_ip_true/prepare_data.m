% dimension input
D_X = size(X,1);
nX = size(y(:,order + 1:end),2);

% training structure input, outputdata
[X_exogen,X_exogen_ref,y_output,y_ref_output,scaleshift,D,D_hidden,D_output] = deep_ssarx(X,X_ref,y,y_ref,order,non_rec);

% power and noise
sf = log(exp(1) - 1); 
sn = log(exp(0.1) - 1); 
sf_hidden = log(exp(1) - 1);
sn_hidden = log(exp(0.1) - 1);
sf_output = log(exp(1) - 1);
sn_output = log(exp(0.1) - 1);

% create variable for all optimized parameters
hyp_state.layer1.mu = y_output;
hyp_state.layer1.sigma = log(exp(sqrt(rand(1,size(hyp_state.layer1.mu,2))./3+0.15)) - 1);
for i = 2:layers
    hyp_state.(strcat('layer',num2str(i))).mu =  y_output;
    hyp_state.(strcat('layer',num2str(i))).sigma = log(exp(sqrt(rand(1,size(hyp_state.(strcat('layer',num2str(i))).mu,2))./3+0.15)) - 1);
end%for

% training structure deep gp
[X_inputs,~] = deep_ssarx_input(X_exogen,hyp_state,layers,order,non_rec);

% kernel and means parameters data input-layer
[hyp_input,readed_kernels] = kernel_mean_parameters(X_inputs{1},sn,sf);                     
                                                                                                                                                            
% kernel and means parameters data hidden-layer
[hyp_hidden,readed_kernels_hidden] = kernel_mean_parameters(X_inputs{2},sn_hidden,sf_hidden);                     

% kernel and means parameters data output-layer
[hyp_output,readed_kernels_output] = kernel_mean_parameters(X_inputs{end},sn_output,sf_output);

% create variable for all optimized parameters
hyp.layer1 = hyp_input;
hyp.layer1.mu = hyp_state.layer1.mu;
hyp.layer1.sigma = hyp_state.layer1.sigma;
hyp.layer1.sn = sn;
hyp.layer1.mu_alpha = randn(1,mm);
hyp.layer1.sigma_alpha = log(exp(sqrt(rand([1,mm])./3+0.15)) - 1);
hyp.layer1.mu_spectral = randn(D,mm);
hyp.layer1.sigma_spectral = - 3.5 * ones(D,mm);
hyp.layer1.nat_spectral = normrnd(0,1,D,mm);
hyp.layer1.(readed_kernels).lengthscale_p = log(exp(inf(size(hyp.layer1.(readed_kernels).lengthscale))) - 1);
J = randperm(nX);
I = J(1:mm);
if strcmp(initial_pseudo_X_sub,'on')
    hyp.layer1.Y_M = X_inputs{1}(:,I);
elseif strcmp(initial_pseudo_X_zero,'on')
    hyp.layer1.Y_M = zeros(size(Y{1}(:,I)));
end
hyp.layer1.phase = unifrnd(0,2*pi,1,mm);
for i = 2:layers
    hyp.(strcat('layer',num2str(i))) = hyp_hidden;
    hyp.(strcat('layer',num2str(i))).mu = hyp_state.(strcat('layer',num2str(i))).mu;
    hyp.(strcat('layer',num2str(i))).sigma = hyp_state.(strcat('layer',num2str(i))).sigma;
    hyp.(strcat('layer',num2str(i))).sn = sn_hidden;
    hyp.(strcat('layer',num2str(i))).mu_alpha = randn(1,mm);
    hyp.(strcat('layer',num2str(i))).sigma_alpha = log(exp(sqrt(rand([1,mm])./3+0.15)) - 1);
    hyp.(strcat('layer',num2str(i))).mu_spectral = randn(D_hidden,mm);
    hyp.(strcat('layer',num2str(i))).sigma_spectral = - 3.5 * ones(D_hidden,mm);
    hyp.(strcat('layer',num2str(i))).nat_spectral = normrnd(0,1,D_hidden,mm);
    hyp.(strcat('layer',num2str(i))).(readed_kernels).lengthscale_p = log(exp(inf(size(hyp.(strcat('layer',num2str(i))).(readed_kernels).lengthscale))) - 1);
    if  strcmp(initial_pseudo_X_sub,'on')
        hyp.(strcat('layer',num2str(i))).Y_M = X_inputs{i}(:,I);
    elseif strcmp(initial_pseudo_X_zero,'on')
        hyp.(strcat('layer',num2str(i))).Y_M = zeros(size(Y{i}(:,I)));
    end
    hyp.(strcat('layer',num2str(i))).phase = unifrnd(0,2*pi,1,mm);
end%for
hyp.output = hyp_output;
hyp.output.sn = sn_output;
hyp.('output').mu_alpha = randn(1,mm);
hyp.('output').sigma_alpha = log(exp(sqrt(rand([1,mm])./3+0.15)) - 1);
hyp.('output').mu_spectral = randn(D_output,mm);
hyp.('output').sigma_spectral = - 3.5 * ones(D_output,mm);
hyp.('output').nat_spectral = normrnd(0,1,D_output,mm);
hyp.('output').(readed_kernels).lengthscale_p = log(exp(inf(size(hyp.('output').(readed_kernels).lengthscale))) - 1);
if  strcmp(initial_pseudo_X_sub,'on')
    hyp.('output').Y_M = X_inputs{layers + 1}(:,I);
elseif strcmp(initial_pseudo_X_zero,'on')
    hyp.('output').Y_M = zeros(size(Y{layers + 1}(:,I)));
end
hyp.('output').phase = unifrnd(0,2*pi,1,mm);

% config
config.n_ref = size(y_ref_output,2);
config.nX = nX;
config.nY = nX;
config.D_hidden = D_hidden;
config.D_output = D_output;
config.D_multi = size(y(:,order + 1:end),1);
config.D_X = D_X;
config.D = D;
config.mm = mm;
config.layers = layers;
config.kernel_fun_statistics1 = kernel_fun_statistics1;
config.kernel_fun_statistics2 = kernel_fun_statistics2;
config.non_rec = non_rec;
config.scaleshift = scaleshift;
config.create_data_python = create_data_python;
config.simulation = simulation;
config.filename_optimized_parameters = filename_optimized_parameters;
config.version_variational_spectrum = version_variational_spectrum;
config.readed_kernels = readed_kernels;
config.initial_pseudo_X_sub = initial_pseudo_X_sub;
config.initial_pseudo_X_zero =initial_pseudo_X_zero;
config.param_kernel = param_kernel;
config.hyp = hyp;
config.distance_fun = distance_fun;
config.distance_diag_fun = distance_diag_fun;
config.distance_var_fun = distance_var_fun;
config.order = order;

