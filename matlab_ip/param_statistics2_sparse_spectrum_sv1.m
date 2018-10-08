function [d_n,D_n,Z_n_W,norm_U_W] = param_statistics2_sparse_spectrum_sv1(Mu,Sigma,norm_U_temp,B,inv_B,inv_B_b_bold,b_bold,config)

% automatic relevance detemination (ard)

inv_repmat_Sigma_j_mm = repmat(reshape(1./Sigma,1,1,config.D),config.mm,config.mm);
repmat_Mu_j_mm = repmat(reshape(Mu,1,1,config.D),config.mm,config.mm);
W = B + repmat(reshape(Sigma,1,1,config.D),config.mm,config.mm);
D_n = 1./(inv_B + inv_repmat_Sigma_j_mm);
d_n = D_n .* (inv_B_b_bold + inv_repmat_Sigma_j_mm .* repmat_Mu_j_mm);
index_d_n = isnan(d_n);
d_n(index_d_n) = repmat_Mu_j_mm(index_d_n);
norm_U_W = prod(norm_U_temp./sqrt(W),3);
Z_n_W =  exp(-1/2 * sum((b_bold - repmat_Mu_j_mm).^2 ./ W,3));

end