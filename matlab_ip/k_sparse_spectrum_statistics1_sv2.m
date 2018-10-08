function [K,S_hat] = k_sparse_spectrum_statistics1_sv2(Mu,Sigma,S,Y_M,config)

% squared exponential kernel

hypi = log(1 + exp(config.hyp.(config.readed_kernels).lengthscale));
hypi_p = log(1 + exp(config.hyp.(config.readed_kernels).lengthscale_p));
sfquad = log(1 + exp(config.hyp.(config.readed_kernels).sf)).^2;
S_hat = repmat(1./hypi,config.mm,1) .* S  + repmat(2*pi./hypi_p,config.mm,1);

K = sqrt((2 * sfquad /config.mm)) .* exp(-1/2* sum(repmat(reshape(S_hat.^2,1,config.mm,config.D),config.nX,1,1) .* repmat(reshape(Sigma,config.nX,1,config.D),1,config.mm,1) ,3)) .* cos(sum(repmat(reshape(S_hat,1,config.mm,config.D),config.nX,1,1) .* (repmat(reshape(Mu,config.nX,1,config.D),1,config.mm,1) - repmat(reshape(Y_M,1,config.mm,config.D),config.nX,1,1)),3) + repmat(config.hyp.phase,config.nX,1));

end