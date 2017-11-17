function [K] = k_sparse_spectrum_statistics2_sv2(S_hat,Mu,Sigma,Y_M,config)

% squared exponential kernel

sfquad = log(1 + exp(config.hyp.(config.readed_kernels).sf))^2;
b = repmat(config.hyp.phase,config.mm,1);
S_hat_Y_M = repmat(sum(S_hat .* Y_M,2)',config.mm,1);
S_hat_Y_M_b =  - S_hat_Y_M + b;
big_sum_minus = S_hat_Y_M_b - S_hat_Y_M_b';
big_sum_plus = S_hat_Y_M_b + S_hat_Y_M_b';
repmat_S_hat = repmat(reshape(S_hat,1,config.mm,config.D),config.mm,1,1);
repmat_S_hat_transpose = repmat(reshape(S_hat,config.mm,1,config.D),1,config.mm,1);
S_hat_minus = repmat_S_hat - repmat_S_hat_transpose;
S_hat_plus = repmat_S_hat + repmat_S_hat_transpose;

K = zeros(config.mm,config.mm);
K_temp = zeros(config.mm,config.mm,config.nX);
for j = 1:config.nX    
    repmat_Sigma = repmat(reshape(Sigma(j,:),1,1,config.D),config.mm,config.mm);
    repmat_Mu = repmat(reshape(Mu(j,:),1,1,config.D),config.mm,config.mm);
    K_temp(:,:,j) = exp(-1/2 * sum(S_hat_minus.^2 .* repmat_Sigma,3)) .* cos(sum(S_hat_minus .*repmat_Mu,3) + big_sum_minus) + exp(-1/2 * sum(S_hat_plus.^2 .* repmat_Sigma,3)) .* cos(sum(S_hat_plus .*repmat_Mu,3) + big_sum_plus);
    K = K + K_temp(:,:,j);
end%for

K = (sfquad / config.mm) .*  K;

end