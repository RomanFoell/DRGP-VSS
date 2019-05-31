import numpy as np
from LVMvSSGP_model_SV1_IP import LVMvSSGP
import scipy.io as sio
from time import time
def extend(x, y, z = {}):
    dictx=dict(x.items())
    dicty=dict(y.items())
    dictz=dict(z.items())
    dictx.update(dicty)
    dictx.update(dictz)
    return dictx
pool, global_f, global_g = None, None, None
def eval_f_LL(MU, X, params):
    return global_f['LL'](**extend({'X': X}, params))
def eval_g_LL(name, MU, X, params):
    return global_g[name]['LL'](**extend({'X': X}, params))

class LVMvSSGP_opt():
    def __init__(self, dataset, run, Q, D, N, M, lower_bound_values, save_iter, inputs, opt_params, fixed_params):
        self.dataset = dataset
        self.run = run
        self.LVMvssgp, self.N, self.M, self.fixed_params, self.lower_bound_values, self.save_iter  = LVMvSSGP(Q, D, N, M), N, M, fixed_params, lower_bound_values, save_iter
        self.inputs = inputs
        self.opt_param_names = [n for n,_ in opt_params.items()]
        opt_param_values = [np.atleast_2d(opt_params[n]) for n in self.opt_param_names]
        self.shapes = [v.shape for v in opt_param_values]
        self.sizes = [sum([np.prod(x) for x in self.shapes[:i]]) for i in range(len(self.shapes)+1)]
        self.callback_counter = [0]

    def unpack(self, x):
        x_param_values = [np.squeeze(x[self.sizes[i-1]:self.sizes[i]].reshape(self.shapes[i-1])) for i in range(1,len(self.shapes)+1)]
        params = {n:v for (n,v) in zip(self.opt_param_names, x_param_values)}
        return params

    def func(self, x):
        params = extend(self.fixed_params, self.unpack(x))
        params = extend(self.inputs, params)
        LL, KL = self.LVMvssgp.f['LL'](**params), self.LVMvssgp.f['KL'](**params)
        return -(LL - KL)

    def fprime(self, x):
        grads, params = [], extend(self.fixed_params, self.unpack(x))
        for n in self.opt_param_names:
            params = extend(self.inputs, params)
            dLL, dKL = self.LVMvssgp.g[n]['LL'](**params), self.LVMvssgp.g[n]['KL'](**params)
            grads += [-(dLL - dKL)]
        return np.concatenate([grad.flatten() for grad in grads])

    def callback(self, x):
        opt_params = self.unpack(x)        
        params = extend(self.inputs, self.fixed_params, opt_params)
        if self.lower_bound_values == 1:  
            LL = self.LVMvssgp.f['LL'](**params)
            KL = self.LVMvssgp.f['KL'](**params)
            print(str(self.callback_counter) + '::' + str(-(LL - KL)))

            if self.save_iter == 1:
                sio.savemat('temp_SV1_'  + self.dataset + self.run + str(self.callback_counter) + '.mat', {'opt_params': params, 'bound': str(-(LL - KL))})
            else:
                sio.savemat('temp_SV1' + self.dataset + self.run + '.mat', {'opt_params': params})
        else:
            print(str(self.callback_counter))
            if self.save_iter == 1:
                sio.savemat('temp_SV1_'  + self.dataset + self.run + str(self.callback_counter) + '.mat', {'opt_params': params})
            else:
                sio.savemat('temp_SV1' + self.dataset + self.run + '.mat', {'opt_params': params})
        self.callback_counter[0] += 1
        x = np.concatenate([np.atleast_2d(opt_params[n]).flatten() for n in self.opt_param_names])       
        return x
