function [K] = k_gauss_cos(dquad,deff,config)

% squared exponential kernel

sfquad = log(1 + exp(config.hyp.(config.loop{config.i}).(config.readed_kernels).sf))^2;

K = sfquad * exp(-1/2 * dquad) .* cos(2 * pi * deff);

end