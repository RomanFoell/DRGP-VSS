function [deff] = param_ard_p(X,config)

% automatic relevance detemination (ard)

nX = size(X,2);
hypi = log(1 + exp(config.hyp.(config.loop{config.i}).(config.readed_kernels).lengthscale_p));
repmat_hypi_X = repmat(hypi',1,nX);
hypX = X ./ repmat_hypi_X;
deff = d_eucl_p(hypX);

end