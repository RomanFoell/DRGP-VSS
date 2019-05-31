function [hyp,readed_cell_kernels] = kernel_mean_parameters(X,sn,sf)

% initialize parameterization kernel function
l = log(exp((max(X,[],2) - min(X,[],2))) - 1)';
%     l = log(exp(sqrt(max(X,[],2) - min(X,[],2))) - 1)';
l(l<-1e2) = -1e2;    

readed_cell_kernels = 'kernel_SPECTRUM';
hyp.(readed_cell_kernels).lengthscale = l;
hyp.(readed_cell_kernels).sf = sf;
hyp.sn = sn;

end