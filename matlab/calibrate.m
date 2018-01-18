%##############################
% CALIBRATION ##################
%##############################

data_set = 'dryer';

create_data_python = 'off'; % for data creation for python it is not necessary to specify the version
simulation = 'on';

filename_optimized_parameters = 'test_data_dryer_SV1.mat';

order = 5; % different time horizons order for input-data and latent-states
layers = 1; % number of hidden layers (layers+1 Gaussian Prcesses)
mm = 40; % sparse parameters, amount of spectral points and pseudo-inpu points

version_variational_spectrum = 'SV1';  % SV1 (DRGP-VVS), SV2(DRGP-SS)
non_rec = 'off'; % no recurrence first layer

% choose pseudo input points initialization (just one of both can be in or off)
initial_pseudo_X_sub = 'on'; % choose subset as initial pseudo inputs
initial_pseudo_X_zero = 'off';  % choose zeros data for pseudo inputs
