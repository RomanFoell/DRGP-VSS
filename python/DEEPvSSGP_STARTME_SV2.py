# In the most cases it is enough to train ones with fixed sn and sf (b, SIGMA_S were assumed to be always fixed)
# and optional MU_S, U depending on the data-set with about 50 to 100 iterations

owd = 'C:/Users/flo9fe/Desktop/GIT_DRGP_VSS/python'
#owd = '/usr/local/home/foellrn/GIT_DRGP_VSS/python'

import os;
os.chdir(owd)
from DEEPvSSGP_opt_SV2 import DEEPvSSGP_opt
from scipy.optimize import minimize
import scipy.io as sio
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from time import gmtime, strftime

strftime("%Y-%m-%d %H:%M:%S", gmtime())

os.chdir('data')
mat = sio.loadmat('drive100_2_10.mat', squeeze_me=True) # specify filename to load
os.chdir(owd)

non_rec = 0 # choose 1, if your setting in matlab was non_rec = 'on'; otherwise 0

X = mat['X'] # input data, structured
y = mat['y'] # output data
lengthscale = mat['lengthscale'] # lengthscales l
lengthscale_p = mat['lengthscale_p']  # lengthscales p
sn = mat['sn']  # noise parameter
sf = mat['sf']  # power parameter
S = mat['S'] # spectral points
MU = mat['MU'] # variational latent state
SIGMA = mat['SIGMA'] # variational latent state variance
U = np.array(mat['U'], dtype=np.float64) # pseudo input points
b = mat['b'] # phases
D = mat['D'] # input dimensions
layers = mat['layers'] # layers
order = mat['order'] # time horizon

X = np.require(X,dtype=None,requirements='A') 
y = np.require(y,dtype=None,requirements='A') 
lengthscale = np.require(lengthscale,dtype=None,requirements='A')
lengthscale_p = np.require(lengthscale_p,dtype=None,requirements='A')
sn = np.require(sn,dtype=None,requirements='A')
sf = np.require(sf,dtype=None,requirements='A')
S = np.require(S,dtype=None,requirements='A')
MU = np.require(MU,dtype=None,requirements='A')
SIGMA = np.require(SIGMA,dtype=None,requirements='A')
U = np.require(U,dtype=None,requirements='A')
b = np.require(b,dtype=None,requirements='A')
D = np.require(D,dtype=None,requirements='A')
layers = np.require(layers,dtype=None,requirements='A')
order = np.require(order,dtype=None,requirements='A')

Q = X.ndim # input data dimension
N = np.int64(X.shape[0]) # amount of structured data (N hat)
M = np.int64(S.shape[0]) # sparse parameter
D = np.array(D,dtype='int64')
layers = np.int64(layers)
order = np.int64(order)
D_cum_sum = np.array([0])
for i in range(0, layers + 1):
   D_temp = D_cum_sum[-1] + D[i]
   D_cum_sum = np.hstack((D_cum_sum,D_temp))
D_cum_sum = np.array(D_cum_sum,dtype='int64')

D_sum = D.sum()
# S = np.random.normal(0, 1,(M,D_sum))
# U = np.zeros((M,D_sum))
# U = np.random.normal(0, 1,(M,D_sum))

hyp = np.zeros((D_sum,2))
hyp[:,0] = lengthscale
hyp[:,1] = lengthscale_p

lower_bound_values = 0 # show lower bound value for every iteration, less fast
save_iter = 0 # save opt_params every iteration (outcome in \DRGP_VSS\python\...)

opt_params = {'hyp': hyp, 'MU': MU, 'SIGMA': SIGMA} # optimized parameters
fixed_params = {'b': b, 'sn': sn, 'sf': sf, 'S': S, 'U': U} # other not optimized parameters
inputs = {'X': X, 'y': y} # input and output data
DEEPvSSGP_opt1 = DEEPvSSGP_opt(Q, D, layers, order, D_cum_sum, N, M, non_rec, lower_bound_values, save_iter, inputs, opt_params, fixed_params)

# LBFGS
x0 = np.concatenate([np.atleast_2d(opt_params[n]).flatten() for n in DEEPvSSGP_opt1.opt_param_names])
DEEPvSSGP_opt1.callback(x0)
res = minimize(DEEPvSSGP_opt1.func, x0, method='L-BFGS-B', jac=DEEPvSSGP_opt1.fprime,
        options={'ftol': 0, 'disp': False, 'maxiter': 70}, tol=0, callback=DEEPvSSGP_opt1.callback)

opt_param_names = [n for n,_ in opt_params.iteritems()]
opt_param_values = [np.atleast_2d(opt_params[n]) for n in opt_param_names]
shapes = [v.shape for v in opt_param_values]
sizes = [sum([np.prod(x) for x in shapes[:i]]) for i in xrange(len(shapes)+1)]
x_param_values = [res.x[sizes[i-1]:sizes[i]].reshape(shapes[i-1]) for i in xrange(1,len(shapes)+1)]
opt_params = {n:v for (n,v) in zip(opt_param_names, x_param_values)}
opt_params1 = opt_params

opt_params1.update(fixed_params)

os.chdir("..")
sio.savemat('matlab/data_optimized/DRGP_SS_drive.mat', {'opt_params': opt_params1})
