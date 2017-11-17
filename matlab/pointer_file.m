% initialize parameterization kernel function
param_kernel = @param_ard;
        
% initialize distance_function
distance_fun = @d_eucl;
distance_diag_fun = @d_diag_eucl;
distance_var_fun = @d_var_eucl;

% initialize kernel function
if strcmp(version_variational_spectrum,'SV1')
    kernel_fun_statistics1 = @k_sparse_spectrum_statistics1_sv1;
    kernel_fun_statistics2 = @k_sparse_spectrum_statistics2_sv1;
elseif strcmp(version_variational_spectrum,'SV2')
    kernel_fun_statistics1 = @k_sparse_spectrum_statistics1_sv2;
    kernel_fun_statistics2 = @k_sparse_spectrum_statistics2_sv2;
end%if







