# In the most cases it is enough to train ones with fixed sn and sf (b were assumed to be always fixed)
# and optional S, U depending on the data-set with about 50 to 100 iterations

owd = 'C:/Users/flo9fe/Desktop/GIT_IP/python_ip'

import os;
os.chdir(owd)
from LVMvSSGP_opt_SV2_IP import LVMvSSGP_opt
from scipy.optimize import minimize
import scipy.io as sio
import numpy as np
import random
np.set_printoptions(precision=2, suppress=True)
from time import gmtime, strftime, time

strftime("%Y-%m-%d %H:%M:%S", gmtime())

os.chdir('data')
dataset = 'data_usps'
run = input('Enter something: ')
mat = sio.loadmat(dataset + '.mat', squeeze_me=True) # specify filename to load
os.chdir(owd)

X = mat['X'] # output data, structured
lengthscale = mat['lengthscale'] # lengthscales l
lengthscale_p = mat['lengthscale_p']  # lengthscales p
sn = mat['sn']  # noise parameter
sf = mat['sf']  # power parameter
S = mat['S'] # spectral points
MU = mat['MU'] # variational latent state
SIGMA = mat['SIGMA'] # variational latent state variance
U = np.array(mat['U'], dtype=np.float64) # pseudo input points
b = mat['b'] # phases

# eliminate bad matlab to python coversion
X = np.require(X,dtype=None,requirements='A') 
lengthscale = np.require(lengthscale,dtype=None,requirements='A')
lengthscale_p = np.require(lengthscale_p,dtype=None,requirements='A')
sn = np.require(sn,dtype=None,requirements='A')
sf = np.require(sf,dtype=None,requirements='A')
S = np.require(S,dtype=None,requirements='A')
MU = np.require(MU,dtype=None,requirements='A')
SIGMA = np.require(SIGMA,dtype=None,requirements='A')
U = np.require(U,dtype=None,requirements='A')
b = np.require(b,dtype=None,requirements='A')

Q = MU.shape[1] # input data dimension
(N,D) = X.shape # output data dimension
M = U.shape[0]

# S = np.random.normal(0, 1,(M,Q))
#U = np.zeros((M,Q))
#U = np.random.normal(0, 1,(M,Q))
rand_M = random.sample(range(1, 1000), 50)
U = MU[rand_M,:]

hyp = np.zeros((Q,2))
hyp[:,0] = lengthscale
hyp[:,1] = lengthscale_p

lower_bound_values = 1 # show lower bound value for every iteration, less fast
save_iter = 1 # save opt_params every iteration (outcome in \DRGP_VSS\python\...)

opt_params = {'sn': sn, 'sf': sf, 'hyp': hyp,  'U': U, 'S': S, 'MU': MU, 'SIGMA': SIGMA} # optimized parameters
fixed_params = {'b': b,} # other not optimized parameters
inputs = {'X': X} # output data
LVMvSSGP_opt1 = LVMvSSGP_opt(dataset, run, Q, D, N, M, lower_bound_values, save_iter, inputs, opt_params, fixed_params)

# LBFGS
x0 = np.concatenate([np.atleast_2d(opt_params[n]).flatten() for n in LVMvSSGP_opt1.opt_param_names])
LVMvSSGP_opt1.callback(x0)
startTime = time()
#bnds = np.transpose(np.squeeze(np.stack((-25*np.ones((x0.shape[0],1)),25*np.ones((x0.shape[0],1)))),axis=2))
res = minimize(LVMvSSGP_opt1.func, x0, method='L-BFGS-B', jac=LVMvSSGP_opt1.fprime,
        options={'ftol': 0, 'disp': False, 'maxiter': 1000}, tol=0, callback=LVMvSSGP_opt1.callback)

opt_param_names = [n for n,_ in opt_params.items()]
opt_param_values = [np.atleast_2d(opt_params[n]) for n in opt_param_names]
shapes = [v.shape for v in opt_param_values]
sizes = [sum([np.prod(x) for x in shapes[:i]]) for i in range(len(shapes)+1)]
x_param_values = [res.x[sizes[i-1]:sizes[i]].reshape(shapes[i-1]) for i in range(1,len(shapes)+1)]
opt_params = {n:v for (n,v) in zip(opt_param_names, x_param_values)}
opt_params1 = opt_params

opt_params1.update(fixed_params)

sio.savemat('matlab_ip/data_optimized/' + 'SV2_IP_' + dataset + run, {'opt_params': opt_params1})

endTime = time()
print('Running Time: '+str(endTime-startTime))
