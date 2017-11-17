function [K] = k_sparse_spectrum_statistics2_sv1(Mu_S_hat,Mu,Sigma,Sigma_S,Y_M,config)

% squared exponential kernel

hypi = log(1 + exp(config.hyp.(config.readed_kernels).lengthscale));
hypi_quad = (log(1 + exp(config.hyp.(config.readed_kernels).lengthscale)).^2);
sfquad = log(1 + exp(config.hyp.(config.readed_kernels).sf))^2;
b = repmat(config.hyp.phase',1,config.mm)';
Mu_S_hat_Y_M = repmat(sum(Mu_S_hat .* Y_M,2)',config.mm,1);
Mu_S_hat_Y_M_b =  -Mu_S_hat_Y_M + b;
big_sum_minus = Mu_S_hat_Y_M_b - Mu_S_hat_Y_M_b';
big_sum_plus = Mu_S_hat_Y_M_b + Mu_S_hat_Y_M_b';
repmat_Mu_S_hat = repmat(reshape(Mu_S_hat,1,config.mm,config.D),config.mm,1,1);
repmat_Mu_S_hat_transpose =  repmat(reshape(Mu_S_hat,config.mm,1,config.D),1,config.mm,1);

Mu_S_hat_minus = repmat_Mu_S_hat - repmat_Mu_S_hat_transpose;
Mu_S_hat_plus = repmat_Mu_S_hat + repmat_Mu_S_hat_transpose;
repmat_hypi_S = repmat(reshape(sqrt(repmat(hypi_quad,config.mm,1)) ./ sqrt(Sigma_S),1,config.mm,config.D),config.mm,1,1);

repmat_hypi_quad_mm = repmat(reshape(hypi_quad,1,1,config.D),config.mm,config.mm);
repmat_Sigma_S = repmat(reshape(Sigma_S,1,config.mm,config.D),config.mm,1,1);
repmat_Sigma_S_transpose = permute(repmat_Sigma_S,[2 1 3]);
sum_Sigma_S_Y_M = repmat(reshape(Sigma_S .* Y_M,1,config.mm,config.D),config.mm,1,1);
repmat_Y_M = repmat(reshape(Y_M,1,config.mm,config.D),config.mm,1,1);

b_bold_denomi = repmat_Sigma_S + repmat_Sigma_S_transpose;
b_bold = (sum_Sigma_S_Y_M + permute(sum_Sigma_S_Y_M,[2 1 3])) ./ b_bold_denomi;
B = repmat_hypi_quad_mm./ b_bold_denomi;
inv_B = 1./B;
inv_B_b_bold = inv_B .* b_bold;
U = repmat_hypi_quad_mm .* (1 ./ repmat_Sigma_S + 1 ./ repmat_Sigma_S_transpose);
norm_U_temp = (repmat_hypi_S .* permute(repmat_hypi_S,[2 1 3])) ./ sqrt(U); 
Z_n_U = exp(-1/2 * sum((repmat_Y_M - permute(repmat_Y_M,[2 1 3])).^2./U,3));

[C,zeta,Z,norm] = param_statistics2_sparse_spectrum_sv1_diag(Mu,Sigma,Sigma_S,Y_M,hypi,config);
repmat_MU_S_hat_N = repmat(reshape(Mu_S_hat,1,config.mm,config.D),config.nX,1,1);
b_N = repmat(config.hyp.phase,config.nX,1);
K_diag = sum(sqrt(norm) .* Z .* exp(-2 * sum(repmat_MU_S_hat_N.^2 .* C,3)) .* cos(2 * (sum(repmat_MU_S_hat_N .* zeta,3) + b_N)),1);

K = zeros(config.mm,config.mm);
K_temp = zeros(config.mm,config.mm,config.nX);
for j = 1:config.nX
    [d_n,D_n,Z_n_W,norm_U_W] = param_statistics2_sparse_spectrum_sv1(Mu(j,:),Sigma(j,:),norm_U_temp,B,inv_B,inv_B_b_bold,b_bold,config);
    K_temp(:,:,j) = Z_n_W .* norm_U_W .*  (exp(-1/2 * sum(Mu_S_hat_minus.^2 .* D_n,3)) .* cos(sum(Mu_S_hat_minus .*d_n,3) + big_sum_minus) + exp(-1/2 * sum(Mu_S_hat_plus.^2 .* D_n,3)) .* cos(sum(Mu_S_hat_plus .*d_n,3) + big_sum_plus));
    K = K + K_temp(:,:,j);
end%for

K = (sfquad / config.mm) .* Z_n_U .*  K;

K_diag = (sfquad / config.mm) .* (config.nX + K_diag);

K = K - diag(diag(K)) + diag(K_diag);

end
