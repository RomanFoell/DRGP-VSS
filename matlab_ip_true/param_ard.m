function [dquad] = param_ard(X,config)

% automatic relevance detemination (ard)

nX = size(X,2);
hypi = log(1 + exp(config.hyp.(config.loop{config.i}).(config.readed_kernels).lengthscale));
repmat_hypi_X = repmat(hypi',1,nX);
hypX = X ./ repmat_hypi_X;
dquad = d_eucl(hypX);

end