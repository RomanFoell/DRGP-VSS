import numpy as np
from DEEPvSSGP_model_SV1 import DEEPvSSGP
import scipy.io as sio
def extend(x, y, z = {}):
    return dict(x.items() + y.items() + z.items())
pool, global_f, global_g = None, None, None
def eval_f_LL(MU, X, params):
    return global_f['LL'](**extend({'X': X}, params))
def eval_g_LL(name, MU, X, params):
    return global_g[name]['LL'](**extend({'X': X}, params))

class DEEPvSSGP_opt():
    def __init__(self, Q, D, layers, order, D_cum_sum, N, M, non_rec, lower_bound_values, save_iter, inputs, opt_params, fixed_params):
        self.deepvssgp, self.D_cum_sum, self.N, self.M, self.fixed_params, self.lower_bound_values, self.save_iter = DEEPvSSGP(Q, D, layers, order, D_cum_sum, N, M, non_rec), D_cum_sum, N, M, fixed_params, lower_bound_values, save_iter
        self.inputs = inputs
        self.opt_param_names = [n for n,_ in opt_params.iteritems()]
        opt_param_values = [np.atleast_2d(opt_params[n]) for n in self.opt_param_names]
        self.shapes = [v.shape for v in opt_param_values]
        self.sizes = [sum([np.prod(x) for x in self.shapes[:i]]) for i in xrange(len(self.shapes)+1)]
        self.callback_counter = [0]

    def unpack(self, x):
        x_param_values = [np.squeeze(x[self.sizes[i-1]:self.sizes[i]].reshape(self.shapes[i-1])) for i in xrange(1,len(self.shapes)+1)]
        params = {n:v for (n,v) in zip(self.opt_param_names, x_param_values)}
        return params

    def func(self, x):
        params = extend(self.fixed_params, self.unpack(x))
        params = extend(self.inputs, params)
        LL, KL = self.deepvssgp.f['LL'](**params), self.deepvssgp.f['KL'](**params)
        return -(LL - KL)

    def fprime(self, x):
        grads, params = [], extend(self.fixed_params, self.unpack(x))
        for n in self.opt_param_names:
            params = extend(self.inputs, params)
            dLL, dKL = self.deepvssgp.g[n]['LL'](**params), self.deepvssgp.g[n]['KL'](**params)
            grads += [-(dLL - dKL)]
        return np.concatenate([grad.flatten() for grad in grads])

    def callback(self, x):
        opt_params = self.unpack(x)        
        params = extend(self.inputs, self.fixed_params, opt_params)
        if self.lower_bound_values == 1:  
            LL = self.deepvssgp.f['LL'](**params)
            KL = self.deepvssgp.f['KL'](**params)
            print(str(self.callback_counter) + ':' + str(-(LL - KL)))
            if self.save_iter == 1:
                sio.savemat('temp_SV1_' + str(self.callback_counter) + '.mat', {'opt_params': params, 'bound': str(-(LL - KL))})
            else:
                sio.savemat('temp_SV1.mat', {'opt_params': params})
        else:
            print(str(self.callback_counter))
            if self.save_iter == 1:
                sio.savemat('temp_SV1_' + str(self.callback_counter) + '.mat', {'opt_params': params})
            else:
                sio.savemat('temp_SV1.mat', {'opt_params': params})
        self.callback_counter[0] += 1
        x = np.concatenate([np.atleast_2d(opt_params[n]).flatten() for n in self.opt_param_names])    
        return x
