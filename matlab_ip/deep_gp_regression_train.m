function [B,opt_hyp] = deep_gp_regression_train(X,y,config)

if strcmp(config.create_data_python,'on')

D = config.D;
for i = 1:config.layers-1
    D = [D config.D_hidden];
end%for
D = [D config.D_output];

SIGMA_S = [];
MU_S = [];
S  = [];
MU_A = [];
SIGMA_A = [];
MU = [];
SIGMA = [];
sn = [];
lengthscale = [];
sf = [];
b = [];
U = [];
for i = 1:config.layers
    MU_A = [MU_A,config.hyp.(strcat('layer',num2str(i))).mu_alpha'];
    SIGMA_A = [SIGMA_A,config.hyp.(strcat('layer',num2str(i))).sigma_alpha'];
    SIGMA_S = [SIGMA_S,config.hyp.(strcat('layer',num2str(i))).sigma_spectral'];
    MU_S = [MU_S,config.hyp.(strcat('layer',num2str(i))).mu_spectral'];
    S = [S,config.hyp.(strcat('layer',num2str(i))).nat_spectral'];
    MU = [MU,config.hyp.(strcat('layer',num2str(i))).mu'];
    SIGMA = [SIGMA,config.hyp.(strcat('layer',num2str(i))).sigma'];
    sf = [sf,config.hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.sf];
    sn = [sn,config.hyp.(strcat('layer',num2str(i))).sn];
    lengthscale = [lengthscale;config.hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.lengthscale'];
    b = [b,config.hyp.(strcat('layer',num2str(i))).phase'];
    U = [U,config.hyp.(strcat('layer',num2str(i))).Y_M'];
end%for
MU_A = [MU_A,config.hyp.('output').mu_alpha'];
SIGMA_A = [SIGMA_A,config.hyp.('output').sigma_alpha'];
SIGMA_S = [SIGMA_S,config.hyp.('output').sigma_spectral'];
MU_S = [MU_S,config.hyp.('output').mu_spectral'];
S = [S,config.hyp.('output').nat_spectral'];
sf = [sf,config.hyp.('output').kernel_SPECTRUM.sf];
sn = [sn,config.hyp.('output').sn];
lengthscale = [lengthscale;config.hyp.('output').kernel_SPECTRUM.lengthscale'];
b = [b,config.hyp.('output').phase'];
U = [U,config.hyp.('output').Y_M'];
X = X';
layers = config.layers;
order =config.order;
lengthscale_p = linspace(1e32,1e32,sum(D))';

cd ..

save('python_IP/data/load_python.mat','X','y','lengthscale','lengthscale_p','sn','sf','S','MU_S','SIGMA_S','MU','SIGMA','MU_A','SIGMA_A','U','b','D','layers','order')

B = [];
opt_hyp = [];

elseif strcmp(config.simulation,'on')

    jitter = 1e-6;
    if strcmp(config.version_variational_spectrum,'SV1')
        config.kernel_fun_statistics1 = @k_sparse_spectrum_statistics1_sv1;
        config.kernel_fun_statistics2 = @k_sparse_spectrum_statistics2_sv1;
    elseif strcmp(config.version_variational_spectrum,'SV2')
        config.kernel_fun_statistics1 = @k_sparse_spectrum_statistics1_sv2;
        config.kernel_fun_statistics2 = @k_sparse_spectrum_statistics2_sv2;
    end%if
    cd data_optimized
    load(config.filename_optimized_parameters)
    cd ..
    opt_hyp = config.hyp;

    if exist('opt_params','var')
        D_temp = config.D;
        for i = 1:config.layers-1
            D_temp = [D_temp config.D_hidden];
        end%for
        D_temp = [D_temp config.D_output];

        D_sum = 0;
        for i = 1:config.layers + 1
            D_temp1 = D_sum(end) + D_temp(i);
            D_sum = [D_sum D_temp1];
        end%for
        for i = 1:config.layers
            if isfield(opt_params,'SIGMA_S')
                opt_hyp.(strcat('layer',num2str(i))).sigma_spectral = double(opt_params.SIGMA_S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if isfield(opt_params,'MU_S')
                opt_hyp.(strcat('layer',num2str(i))).mu_spectral = double(opt_params.MU_S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if isfield(opt_params,'S')
                opt_hyp.(strcat('layer',num2str(i))).nat_spectral = double(opt_params.S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if config.layers == 1
                opt_hyp.(strcat('layer',num2str(i))).mu = double(opt_params.MU(i,:));
                opt_hyp.(strcat('layer',num2str(i))).sigma = double(opt_params.SIGMA(i,:));
            else
                opt_hyp.(strcat('layer',num2str(i))).mu = double(opt_params.MU(:,i)');
                opt_hyp.(strcat('layer',num2str(i))).sigma = double(opt_params.SIGMA(:,i)');
            end%if
            opt_hyp.(strcat('layer',num2str(i))).sn = double(opt_params.sn(i));
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.lengthscale = double(opt_params.hyp(D_sum(i) + 1:D_sum(i+1),1)');
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.lengthscale_p = double(opt_params.hyp(D_sum(i) + 1:D_sum(i+1),2)');
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.sf = double(opt_params.sf(i));
            opt_hyp.(strcat('layer',num2str(i))).Y_M = double(opt_params.U(:,D_sum(i) + 1:D_sum(i+1))');
            opt_hyp.(strcat('layer',num2str(i))).phase = double(opt_params.b(:,i)');
        end%for
        if isfield(opt_params,'SIGMA_S')
            opt_hyp.('output').sigma_spectral = double(opt_params.SIGMA_S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        if isfield(opt_params,'MU_S')
            opt_hyp.('output').mu_spectral = double(opt_params.MU_S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        if isfield(opt_params,'S')
            opt_hyp.('output').nat_spectral = double(opt_params.S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        opt_hyp.('output').sn = opt_params.sn(i+1);
        opt_hyp.('output').kernel_SPECTRUM.lengthscale = double(opt_params.hyp(D_sum(i+1) + 1:D_sum(i+2),1)');
        opt_hyp.('output').kernel_SPECTRUM.lengthscale_p = double(opt_params.hyp(D_sum(i+1) + 1:D_sum(i+2),2)');
        opt_hyp.('output').kernel_SPECTRUM.sf = double(opt_params.sf(i+1));
        opt_hyp.('output').Y_M = double(opt_params.U(:,D_sum(i+1) + 1:D_sum(i+2))');
        opt_hyp.('output').phase = double(opt_params.b(:,i+1)');
    else
        D_temp = config.D;
        for i = 1:config.layers-1
            D_temp = [D_temp config.D_hidden];
        end%for
        D_temp = [D_temp config.D_output];

        D_sum = 0;
        for i = 1:config.layers + 1
            D_temp1 = D_sum(end) + D_temp(i);
            D_sum = [D_sum D_temp1];
        end%for
        for i = 1:config.layers
            if exist('SIGMA_S')
                opt_hyp.(strcat('layer',num2str(i))).sigma_spectral = double(SIGMA_S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if exist('MU_S')
                opt_hyp.(strcat('layer',num2str(i))).mu_spectral = double(MU_S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if exist('S')
                opt_hyp.(strcat('layer',num2str(i))).nat_spectral = double(S(:,D_sum(i) + 1:D_sum(i+1))');
            end%if
            if config.layers == 1
                opt_hyp.(strcat('layer',num2str(i))).mu = double(MU');
                opt_hyp.(strcat('layer',num2str(i))).sigma = double(SIGMA');
            else
                opt_hyp.(strcat('layer',num2str(i))).mu = double(MU(:,i)');
                opt_hyp.(strcat('layer',num2str(i))).sigma = double(SIGMA(:,i)');
            end%if
            opt_hyp.(strcat('layer',num2str(i))).sn = double(sn(i));
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.lengthscale = double(lengthscale(D_sum(i) + 1:D_sum(i+1)));
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.lengthscale_p = double(lengthscale_p(D_sum(i) + 1:D_sum(i+1)));
            opt_hyp.(strcat('layer',num2str(i))).kernel_SPECTRUM.sf = double(sf(i));
            opt_hyp.(strcat('layer',num2str(i))).Y_M = double(U(:,D_sum(i) + 1:D_sum(i+1))');
            opt_hyp.(strcat('layer',num2str(i))).phase = double(b(:,i)');
        end%for
        if exist('SIGMA_S')
            opt_hyp.('output').sigma_spectral = double(SIGMA_S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        if exist('MU_S')
            opt_hyp.('output').mu_spectral = double(MU_S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        if exist('S')
            opt_hyp.('output').nat_spectral = double(S(:,D_sum(i+1) + 1:D_sum(i+2))');
        end%if
        opt_hyp.('output').sn = sn(i+1);
        opt_hyp.('output').kernel_SPECTRUM.lengthscale = double(lengthscale(D_sum(i+1) + 1:D_sum(i+2)));
        opt_hyp.('output').kernel_SPECTRUM.lengthscale_p = double(lengthscale_p(D_sum(i+1) + 1:D_sum(i+2)));
        opt_hyp.('output').kernel_SPECTRUM.sf = double(sf(i+1));
        opt_hyp.('output').Y_M = double(U(:,D_sum(i+1) + 1:D_sum(i+2))');
        opt_hyp.('output').phase = double(b(:,i+1)');
        X = X';
    end%if


    color_order = [0         0.4470    0.7410
                   0.8500    0.3250    0.0980
                   0.9290    0.6940    0.1250
                   0.4940    0.1840    0.5560
                   0.4660    0.6740    0.1880
                   0.3010    0.7450    0.9330
                   0.6350    0.0780    0.1840];
    color_order_var = [127 164 189
                       218 167 146
                       238 215 160
                       135 96  143
                       147 172 115
                       160 215 238
                       164 110 119]/255;
    figure
    for l = 1:config.layers
        fill([1:size(opt_hyp.(strcat('layer',num2str(l))).sigma,2) fliplr(1:size(opt_hyp.(strcat('layer',num2str(l))).sigma,2))],[opt_hyp.(strcat('layer',num2str(l))).mu + 2*sqrt(log(1 + exp(opt_hyp.(strcat('layer',num2str(l))).sigma).^2)),fliplr(opt_hyp.(strcat('layer',num2str(l))).mu - 2*sqrt(log(1 + exp(opt_hyp.(strcat('layer',num2str(l))).sigma).^2)))],color_order_var(l,:),'edgecolor',color_order_var(l,:),'DisplayName','2 times SD');
        hold on
    end%for
    for l = 1:config.layers
        plot(1:size(opt_hyp.(strcat('layer',num2str(l))).mu,2),opt_hyp.(strcat('layer',num2str(l))).mu,'DisplayName',strcat('state',num2str(l)),'color',color_order(l,:));
        hold on
    end%for
    title('learned hidden states')
    legend('show')

    [X_inputs,SIGMA_inputs] = deep_ssarx_input(X,opt_hyp,config.layers,config.order,config.non_rec);

    % not needed in loop
    k_config.nX = config.nX;
    k_config.mm = config.mm;
    k_config.readed_kernels = config.readed_kernels;
    if exist('MEAN_MAP')
        %
    else
        opt_params.MEAN_MAP = zeros(1,size(X,1));
    end%

    % all layers
    config.loop = fieldnames(opt_hyp);
    B_alpha = cell(config.layers + 1,1);
    B_trace =  cell(config.layers + 1,1);
    B_invK_MM = cell(config.layers + 1,1);
    L = cell(config.layers + 1,1);
    A_K_MN_NM_chol = cell(config.layers + 1,1);
    statistics1 = cell(config.layers + 1,1);
    for i = 1:config.layers+1
        snquad = log(1 + exp(opt_hyp.(config.loop{i}).sn)).^2;
        SIGMA_inputs_quad = (log(1 + exp(SIGMA_inputs{i})).^2)';
        Y_M_temp = (opt_hyp.(config.loop{i}).Y_M)';
        if strcmp(config.version_variational_spectrum,'SV1')
            MU_S = (opt_hyp.(config.loop{i}).mu_spectral)';
            SIGMA_S = (log(1 + exp(opt_hyp.(config.loop{i}).sigma_spectral)).^2)';
        elseif  strcmp(config.version_variational_spectrum,'SV2')
            S = (opt_hyp.(config.loop{i}).nat_spectral)';
        end%if
        X_input_temp = X_inputs{i}';
        config_MM.i = i;
        config_MM.loop = config.loop;
        config_MM.readed_kernels = config.readed_kernels;
        config_MM.hyp = opt_hyp;
        config_MM.nX = config.nX;
        config_MM.mm = config.mm;
        dquad_MM = param_ard(Y_M_temp',config_MM);
        deff_MM = param_ard_p(Y_M_temp',config_MM);

        K_MM = k_gauss_cos(dquad_MM,deff_MM,config_MM);

        k_config.D = D_temp(i);
        k_config.hyp = opt_hyp.(config.loop{i});
        if strcmp(config.version_variational_spectrum,'SV1')
            [statistics1{i},MU_S_hat] = feval(config.kernel_fun_statistics1,X_input_temp,SIGMA_inputs_quad,SIGMA_S,MU_S,Y_M_temp,k_config);
            statistics2 = feval(config.kernel_fun_statistics2,MU_S_hat,X_input_temp,SIGMA_inputs_quad,SIGMA_S,Y_M_temp,k_config);
        elseif  strcmp(config.version_variational_spectrum,'SV2')
            [statistics1{i},S_hat] = feval(config.kernel_fun_statistics1,X_input_temp,SIGMA_inputs_quad,S,Y_M_temp,k_config);
            statistics2 = feval(config.kernel_fun_statistics2,S_hat,X_input_temp,SIGMA_inputs_quad,Y_M_temp,k_config);
        end%if

        L{i} = chol(K_MM + jitter * eye(config.mm))';
        A_K_MN_NM_chol{i} = chol(statistics2 + snquad * K_MM + jitter * eye(config.mm))';
        inv_K_MN_NM = solve_chol(A_K_MN_NM_chol{i},eye(config.mm));
        invK_MM = solve_chol(L{i},eye(config.mm));

        if i < config.layers + 1
            MU = opt_hyp.(config.loop{i}).mu;
            if exist('opt_params')
                B_alpha{i} = solve_chol(A_K_MN_NM_chol{i},statistics1{i}') * (MU(config.order+1:end)'- (opt_params.MEAN_MAP * X)');
            else
                B_alpha{i} = solve_chol(A_K_MN_NM_chol{i},statistics1{i}') * (MU(config.order+1:end)'- (MEAN_MAP * X)');
            end%if
        else
            B_alpha{i} = solve_chol(A_K_MN_NM_chol{i},statistics1{i}') * y(:,config.order + 1:end)';
        end%if
         B_trace{i} = snquad * inv_K_MN_NM;
         B_invK_MM{i} = invK_MM;
    end%if

    B.alpha = B_alpha;
    B.trace = B_trace;
    B.invK_MM = B_invK_MM;
end%if

end