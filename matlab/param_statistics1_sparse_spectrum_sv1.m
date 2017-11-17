function [Mu_S_hat,C,zeta,Z,norm] = param_statistics1_sparse_spectrum_sv1(Mu,Sigma,Sigma_S,Mu_S,Y_M,hypi,hypi_p,config)

% automatic relevance detemination (ard)

repmat_hypi_quad_mm = repmat(hypi.^2,config.mm,1);
Sigma_S_hypi_quad = Sigma_S ./ repmat_hypi_quad_mm;
hypi_quad_Sigma_S = 1./Sigma_S_hypi_quad;
repmat_cov_Z =  repmat(reshape(hypi_quad_Sigma_S,1,config.mm,config.D),config.nX,1,1)  + repmat(reshape(Sigma,config.nX,1,config.D),1,config.mm,1);
repmat_Y_M = repmat(reshape(Y_M,1,config.mm,config.D),config.nX,1,1);
repmat_Mu = repmat(reshape(Mu,config.nX,1,config.D),1,config.mm,1);
norm = prod(repmat(reshape(hypi_quad_Sigma_S,1,config.mm,config.D),config.nX,1,1)./repmat_cov_Z,3);
C = 1./(repmat(reshape(Sigma_S_hypi_quad,1,config.mm,config.D),config.nX,1,1) + repmat(reshape((1./Sigma),config.nX,1,config.D),1,config.mm,1));
zeta_temp = C .* (repmat(reshape(Sigma_S_hypi_quad .* Y_M,1,config.mm,config.D),config.nX,1,1) + repmat(reshape((Mu./Sigma),config.nX,1,config.D),1,config.mm,1));
index_zeta_temp = isnan(zeta_temp);
zeta_temp(index_zeta_temp) = repmat_Mu(index_zeta_temp);
zeta = zeta_temp - repmat_Y_M;
Z = exp(-1/2 * sum((repmat_Y_M - repmat_Mu).^2 ./ repmat_cov_Z,3));
Mu_S_hat = repmat(1./hypi,config.mm,1) .* Mu_S  + repmat(2*pi./hypi_p,config.mm,1);

end
