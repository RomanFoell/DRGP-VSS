function [y_test,lambda_test,V_test] = deep_gp_regression_test(X_test,B,config)

n_test = 1;
D_temp = config.D;
for i = 1:config.layers-1
    D_temp = [D_temp config.D_hidden];
end%for
D_temp = [D_temp config.D_output];

% not needed in loop
k_config.nX = n_test;
k_config.mm = config.mm;
k_config.readed_kernels = config.readed_kernels;

y_test = cell(config.layers + 1,config.n_ref);
lambda_test = cell(config.layers + 1,config.n_ref);
V_test = cell(config.layers + 1,config.n_ref);

% init first point
X_simu = cell(config.layers + 1,1);
SIGMA_simu = cell(config.layers + 1,1);
if strcmp(config.non_rec,'on')
    X_simu{1} = [X_test(:,1)]';
    SIGMA_simu{1} = [log(exp(sqrt(zeros(config.order * config.D_X,1))) - 1)]';
else
    X_simu{1} = [X_test(:,1);fliplr(config.hyp.layer1.mu(1,end - config.order + 1:end))']';
    SIGMA_simu{1} = [log(exp(sqrt(zeros(config.order * config.D_X,1))) - 1);fliplr(config.hyp.layer1.sigma(1,end - config.order + 1:end))']';
end%if
config.loop = fieldnames(config.hyp);
for i = 1:config.layers + 1
    snquad = log(1 + exp(config.hyp.(config.loop{i}).sn)).^2;
    SIGMA_inputs_quad = log(1 + exp(SIGMA_simu{i})).^2;
    Y_M_temp = (config.hyp.(config.loop{i}).Y_M)';
    if strcmp(config.version_variational_spectrum,'SV1')
        MU_S = (config.hyp.(config.loop{i}).mu_spectral)';
        SIGMA_S = (log(1 + exp(config.hyp.(config.loop{i}).sigma_spectral)).^2)';
    elseif  strcmp(config.version_variational_spectrum,'SV2')
        S = (config.hyp.(config.loop{i}).nat_spectral)';
    end%if

    k_config.hyp = config.hyp.(config.loop{i});
    k_config.D = D_temp(i);
    if strcmp(config.version_variational_spectrum,'SV1')
        stat0_test = log(1 + exp(config.hyp.(config.loop{i}).(config.readed_kernels).sf)).^2;
        [statistics1_test,MU_S_hat] = feval(config.kernel_fun_statistics1,X_simu{i},SIGMA_inputs_quad,SIGMA_S,MU_S,Y_M_temp,k_config);
        statistics2_test = feval(config.kernel_fun_statistics2,MU_S_hat,X_simu{i},SIGMA_inputs_quad,SIGMA_S,Y_M_temp,k_config);
        statistics2_reg_test = k_sparse_spectrum_statistics2_reg(MU_S,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
    elseif  strcmp(config.version_variational_spectrum,'SV2')
        stat0_test = log(1 + exp(config.hyp.(config.loop{i}).(config.readed_kernels).sf)).^2;
        [statistics1_test,S_hat] = feval(config.kernel_fun_statistics1,X_simu{i},SIGMA_inputs_quad,S,Y_M_temp,k_config);
        statistics2_test = feval(config.kernel_fun_statistics2,S_hat,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
        statistics2_reg_test = k_sparse_spectrum_statistics2_reg(S,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
    end%if
    B_alpha = B.alpha{i};
    B_trace = B.trace{i};
    B_inv_K_MM = B.invK_MM{i};

    % test values
    y_test{i} = statistics1_test * B_alpha;
    lambda_test{i} = B_alpha' * (statistics2_test - statistics1_test' * statistics1_test) * B_alpha + stat0_test - sum(sum(B_inv_K_MM .* statistics2_reg_test)) + snquad;
    V_test{i} = lambda_test{i} + sum(sum(B_trace .* statistics2_test));

    if i < config.layers
        X_simu{i + 1} = [y_test{i,1};fliplr(config.hyp.(strcat('layer',num2str(i))).mu(end - config.order + 1:end - 1))';fliplr(config.hyp.(strcat('layer',num2str(i + 1))).mu(end - config.order + 1:end))']';
        SIGMA_simu{i + 1} = [log(exp(sqrt(V_test{i,1})) - 1);fliplr(config.hyp.(strcat('layer',num2str(i))).sigma(end - config.order + 1:end - 1))';fliplr(config.hyp.(strcat('layer',num2str(i + 1))).sigma(end - config.order + 1:end))']';
    elseif i == config.layers
        X_simu{i + 1} = [y_test{i,1};fliplr(config.hyp.(strcat('layer',num2str(i))).mu(end - config.order + 1:end - 1))']';
        SIGMA_simu{i + 1} = [log(exp(sqrt(V_test{i,1})) - 1);fliplr(config.hyp.(strcat('layer',num2str(i))).sigma(end - config.order + 1:end - 1))']';
    end%if
end%for

% free simulation
for k = 2:config.n_ref
    if strcmp(config.non_rec,'on')
        X_simu{1} = [X_test(:,k)]';
    else
        X_simu{1} = [X_test(:,k)' X_simu{2}(1:config.order)];
    end%if
    if strcmp(config.non_rec,'on')
        SIGMA_simu{1} = [log(exp(sqrt(zeros(config.order * config.D_X,1))) - 1)]';
    else
        SIGMA_simu{1} = [log(exp(sqrt(zeros(config.order * config.D_X,1))) - 1)' SIGMA_simu{2}(1:config.order)];
    end%if
    for i = 1:config.layers + 1
        snquad = log(1 + exp(config.hyp.(config.loop{i}).sn)).^2;
        SIGMA_inputs_quad = log(1 + exp(SIGMA_simu{i})).^2;
        Y_M_temp = (config.hyp.(config.loop{i}).Y_M)';
        if strcmp(config.version_variational_spectrum,'SV1')
            MU_S = (config.hyp.(config.loop{i}).mu_spectral)';
            SIGMA_S = (log(1 + exp(config.hyp.(config.loop{i}).sigma_spectral)).^2)';
        elseif  strcmp(config.version_variational_spectrum,'SV2')
            S = (config.hyp.(config.loop{i}).nat_spectral)';
        end%if

        k_config.D = D_temp(i);
        k_config.hyp = config.hyp.(config.loop{i});
        if strcmp(config.version_variational_spectrum,'SV1')
            stat0_test = log(1 + exp(config.hyp.(config.loop{i}).(config.readed_kernels).sf)).^2;
            [statistics1_test,MU_S_hat] = feval(config.kernel_fun_statistics1,X_simu{i},SIGMA_inputs_quad,SIGMA_S,MU_S,Y_M_temp,k_config);
            statistics2_test = feval(config.kernel_fun_statistics2,MU_S_hat,X_simu{i},SIGMA_inputs_quad,SIGMA_S,Y_M_temp,k_config);
            statistics2_reg_test = k_sparse_spectrum_statistics2_reg(MU_S,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
        elseif  strcmp(config.version_variational_spectrum,'SV2')
            stat0_test = log(1 + exp(config.hyp.(config.loop{i}).(config.readed_kernels).sf)).^2;
            [statistics1_test,S_hat] = feval(config.kernel_fun_statistics1,X_simu{i},SIGMA_inputs_quad,S,Y_M_temp,k_config);
            statistics2_test = feval(config.kernel_fun_statistics2,S_hat,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
            statistics2_reg_test = k_sparse_spectrum_statistics2_reg(S,X_simu{i},SIGMA_inputs_quad,Y_M_temp,k_config);
        end%if
        B_alpha = B.alpha{i};
        B_trace = B.trace{i};
        B_inv_K_MM = B.invK_MM{i};

        % test values
        y_test{i,k} = statistics1_test * B_alpha;
        lambda_test{i,k} = B_alpha' * (statistics2_test - statistics1_test' * statistics1_test) * B_alpha + stat0_test - sum(sum(B_inv_K_MM .* statistics2_reg_test)) + snquad;
        V_test{i,k} = lambda_test{i,k} +  sum(sum(B_trace .* statistics2_test));

        if i < config.layers
            X_simu{i + 1} = [y_test{i,k}' X_simu{i + 1}(1:config.order - 1) X_simu{i + 2}(1:config.order)];
            SIGMA_simu{i + 1} = [log(exp(sqrt(V_test{i,k})) - 1)' SIGMA_simu{i + 1}(1:config.order - 1) SIGMA_simu{i + 2}(1:config.order)];
        elseif i == config.layers
            X_simu{i + 1} = [y_test{i,k}' X_simu{i + 1}(1:config.order - 1)];
            SIGMA_simu{i + 1} = [log(exp(sqrt(V_test{i,k})) - 1)' SIGMA_simu{i + 1}(1:config.order - 1)];
        end%if
    end%for
end%for

end

