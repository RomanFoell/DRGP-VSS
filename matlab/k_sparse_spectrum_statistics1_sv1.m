function [K,Mu_S_hat] = k_sparse_spectrum_statistics1_sv1(Mu,Sigma,Sigma_S,Mu_S,Y_M,config)

% spectrum kernel

hypi = log(1 + exp(config.hyp.(config.readed_kernels).lengthscale));
hypi_p = log(1 + exp(config.hyp.(config.readed_kernels).lengthscale_p));
sfquad = log(1 + exp(config.hyp.(config.readed_kernels).sf))^2;

[Mu_S_hat,C,zeta,Z,norm] = param_statistics1_sparse_spectrum_sv1(Mu,Sigma,Sigma_S,Mu_S,Y_M,hypi,hypi_p,config);

K = sqrt((2 * sfquad * norm /config.mm)) .* Z .* exp(-1/2 * sum(repmat(reshape(Mu_S_hat.^2,1,config.mm,config.D),config.nX,1,1) .* C ,3)) .* cos(sum(repmat(reshape(Mu_S_hat,1,config.mm,config.D),config.nX,1,1) .* zeta,3) + repmat(config.hyp.phase,config.nX,1));

end

