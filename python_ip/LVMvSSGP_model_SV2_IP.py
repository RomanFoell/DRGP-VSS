# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle

print('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.exception_verbosity = 'low'
theano.config.reoptimize_unpickled_function = True

class LVMvSSGP:
    def __init__(self, Q, D, N, M):
        try:
            print('Trying to load model...')
            with open('model_SV2.save', 'rb') as file_handle:
                self.f, self.g = pickle.load(file_handle)
                print('Loaded!')
            return
        except:
            print('Failed. Creating a new model...')

        print('Setting up variables...')
        hyp, S, MU, SIGMA, U, X = T.dmatrices('hyp', 'S', 'MU', 'SIGMA', 'U', 'X')
        b = T.dvector('b')
        sn = T.dscalar('sn')
        sf = T.dscalar('sf') 

        SIGMA_trf = T.log(1+T.exp(SIGMA))**2       
        sf_trf, sn_trf, lengthscale_trf, lengthscale_p_trf  =  T.log(1 + T.exp(sf))**2, T.log(1 + T.exp(sn))**2, T.log(1 + T.exp(hyp[:,0])), T.log(1 + T.exp(hyp[:,1]))
        
        print('Setting up model...')
        LL, KL = self.get_model(lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, S, MU, SIGMA_trf, U, b, X, Q, D, N, M)

        print('Compiling model...')
        
        inputs = {'X': X, 'MU': MU, 'SIGMA': SIGMA, 'S': S, 'U':  U, 'b':  b, 'hyp': hyp, 'sn': sn, 'sf': sf}
        z = 0.0 * sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = {'LL': LL, 'KL': KL}
        self.f = {fn: theano.function(list(inputs.values()), fv+z, name=fn, on_unused_input='ignore') for fn,fv in f.items()}       
        
        g = {'LL': LL, 'KL': KL}
        wrt = {'MU': MU, 'SIGMA': SIGMA, 'S':  S, 'U':  U, 'b':  b, 'hyp': hyp, 'sn': sn, 'sf': sf}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, on_unused_input='ignore') for gn,gv in g.items()} for vn, vv in wrt.items()}

        with open('model_SV2.save', 'wb') as file_handle:
            print('Saving model...')
            sys.setrecursionlimit(100000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def reg_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M, D):
        
#       lengthscale_trf # Q
#       lengthscale_p_trf # Q
#       sf_trf # 1
#       S # M x Q
#       MU # N x Q
#       SIGMA_trf # N x Q
#       U # M x Q
#       b # M
#       N # 1
#       M # 1
        
        b = T.zeros(T.shape(b))
        MU_S = T.zeros(T.shape(S))
        SIGMA_S_trf = T.ones(T.shape(S))
        
        inv_SIGMA_trf = SIGMA_trf**-1 # N x Q
        MU_S_hat = lengthscale_trf**-1 * MU_S + 2 * np.pi * lengthscale_p_trf**-1 # M x Q

        MU_S_hat_U_b =  -(MU_S_hat * U).sum(1)[None,:] + b # M x M
        big_sum_minus = MU_S_hat_U_b - MU_S_hat_U_b.T # M x M
        big_sum_plus = MU_S_hat_U_b + MU_S_hat_U_b.T # M x M
        MU_S_hat_minus = MU_S_hat[None,:,:] - MU_S_hat[:,None,:] # M x M x Q
        MU_S_hat_plus = MU_S_hat[None,:,:] + MU_S_hat[:,None,:] # M x M x Q
        
        u_EEPhiTPhi = (U[None,:,:] - U[:,None,:])**2 # M x M x Q
        b_bold_denomi = SIGMA_S_trf[None,:,:] + SIGMA_S_trf[:,None,:] # M x M x Q
        sum_SIGMA_S_U = SIGMA_S_trf * U # M x Q
        b_bold = (sum_SIGMA_S_U[None,:,:] + sum_SIGMA_S_U[:,None,:])/b_bold_denomi # M x M x Q
        B = (lengthscale_trf**2)[None,None,:]/b_bold_denomi # M x M x Q
        inv_B = 1/B  # M x M x Q
        U_EEPhiTPhi = (lengthscale_trf**2)[None,None,:]*(SIGMA_S_trf[None,:,:]**-1 + SIGMA_S_trf[:,None,:]**-1) # M x M x Q
        norm_EEPhiTPhi_U_temp = lengthscale_trf[None,None,:]**2/((SIGMA_S_trf[None,:,:] * SIGMA_S_trf[:,None,:]) * U_EEPhiTPhi)**0.5 # M x M x Q
        Z_n_U_EEPhiTPhi = np.exp(-0.5 * (u_EEPhiTPhi/U_EEPhiTPhi).sum(2))  # M x M        
        inv_B_b_bold = inv_B * b_bold # M x M x Q
        inv_SIGMA_trf_MU = inv_SIGMA_trf * MU # N x Q
 
        EPhiTPhi = np.zeros((M,M))
        loop = np.int64(-1) 
        
        def EPhiTPhi_loop_i(loop, EPhiTPhi, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp):
            loop = loop + 1
            D_n = (inv_B + inv_SIGMA_trf[loop,:][None,None,:])**-1 # M x M x Q
            d_n = D_n * (inv_B_b_bold + inv_SIGMA_trf_MU[loop,:][None,None,:]) # M x M x Q
            W = B + SIGMA_trf[loop,:][None,None,:]  # M x M x Q
            norm_EEPhiTPhi_U_W = (norm_EEPhiTPhi_U_temp/W**0.5).prod(2) # M x M  % here we put det(U), det(W), because of numeric issues (prod(2) is huge for huge input-dimensions) 
            Z_n_W =  T.exp(-0.5 * ((b_bold - MU[loop,:][None,None,:])**2 / W).sum(2)) # M x M   
            EPhiTPhi = EPhiTPhi + Z_n_W * norm_EEPhiTPhi_U_W * (T.exp(-0.5 * (MU_S_hat_minus**2 * D_n).sum(2)) * T.cos((MU_S_hat_minus * d_n).sum(2) + big_sum_minus) + T.exp(-0.5 * (MU_S_hat_plus**2 * D_n).sum(2)) * T.cos((MU_S_hat_plus * d_n).sum(2) + big_sum_plus))  # M x M
            return loop, EPhiTPhi
        
        result, _ = theano.scan(EPhiTPhi_loop_i,
                            outputs_info = [loop, EPhiTPhi],
                            n_steps = N,
                            non_sequences = [MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp])

        EPhiTPhi_out = result[-1][-1] # M x M
        
        reg_EEPhiTPhi = (sf_trf**2/2) * Z_n_U_EEPhiTPhi * EPhiTPhi_out # M x M

        return reg_EEPhiTPhi

    def get_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M): 
        
#       lengthscale_trf # Q
#       lengthscale_p_trf # Q
#       sf_trf # 1
#       S # M x Q
#       MU # N x Q
#       SIGMA_trf # N x Q
#       U # M x Q
#       b # M
#       N # 1
#       M # 1

        S_hat = lengthscale_trf**-1 * S + 2 * np.pi * lengthscale_p_trf**-1 # M x D     
        decay = T.exp(-0.5 * ((S_hat**2)[None,:, :] * SIGMA_trf[:, None, :]).sum(2)) # N x M x D
        cos_w = T.cos((S_hat[None,:, :] * (MU[:, None, :] - U[None, :, :])).sum(2) + b[None,:]) # N x M x D
        EPhi = (2 * sf_trf/M)**0.5 * decay * cos_w  # N x M

        S_hat_U_b =  -(S_hat * U).sum(1)[None,:] + b # M x M
        big_sum_minus = S_hat_U_b - S_hat_U_b.T # M x M
        big_sum_plus = S_hat_U_b + S_hat_U_b.T # M x M
        S_hat_minus = S_hat[None,:,:] - S_hat[:,None,:] # M x M x Q
        S_hat_plus = S_hat[None,:,:] + S_hat[:,None,:] # M x M x Q 

        EPhiTPhi = np.zeros((M,M))
        loop = np.int64(-1) 
        def EPhiTPhi_loop(loop, EPhiTPhi, MU, SIGMA_trf, S_hat_minus, S_hat_plus, big_sum_minus, big_sum_plus):
            loop = loop + 1
            EPhiTPhi = EPhiTPhi + T.exp(-0.5 * (S_hat_minus**2 * SIGMA_trf[loop,:][None, None, :]).sum(2)) * T.cos((S_hat_minus * MU[loop,:][None, None, :]).sum(2) + big_sum_minus) + T.exp(-0.5 * (S_hat_plus**2 * SIGMA_trf[loop,:][None, None, :]).sum(2)) * T.cos((S_hat_plus * MU[loop,:][None, None, :]).sum(2) + big_sum_plus) # M x M
            return loop, EPhiTPhi
        
        result, _ = theano.scan(EPhiTPhi_loop,
                                outputs_info = [loop, EPhiTPhi],
                                n_steps = N,
                                non_sequences = [MU, SIGMA_trf, S_hat_minus, S_hat_plus, big_sum_minus, big_sum_plus])
        
        EPhiTPhi_out = result[-1][-1] # M x M

        EPhiTPhi = (sf_trf/M) * EPhiTPhi_out  # M x M

        return EPhi, EPhiTPhi
    
    def kernel_gauss(self, U, lengthscale_trf, lengthscale_p_trf, sf_trf):
        
        X_exp = U * lengthscale_trf**-1  # M x Q 
        X_cos = U * lengthscale_p_trf**-1 # M x Q 
        X_exp_product = (X_exp**2).sum(1) # M
        
        dquad = -2 * X_exp.dot(X_exp.T) + (X_exp_product.T)[None,:] + X_exp_product[:,None] # M x M
        dsub = (X_cos[None,:,:] - X_cos[:,None,:]).sum(2) # M x M
        K = sf_trf * T.exp(-1/2 * dquad) * T.cos(2 * np.pi * dsub) # M x M
               
        return K
    

    def get_opt_A(self, sn_trf, EPhiTPhi, XT_EPhi, K_MM):
        cholSigInv = sT.cholesky(EPhiTPhi + sn_trf * K_MM + 1e-6  * T.identity_like(K_MM))
        cholK_MM = sT.cholesky(K_MM + 1e-6 * T.identity_like(K_MM))
        invCholSigInv = sT.matrix_inverse(cholSigInv)
        invCholK_MM = sT.matrix_inverse(cholK_MM)
        InvSig = invCholSigInv.T.dot(invCholSigInv)
        InvK_MM = invCholK_MM.T.dot(invCholK_MM)
        Sig_EPhiT_X = InvSig.dot(XT_EPhi.T)
        return Sig_EPhiT_X, cholSigInv, cholK_MM, InvK_MM

    def get_model(self, lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, S, MU, SIGMA_trf, U, b, X, Q, D, N, M):
                
        EPhi, EPhiTPhi = self.get_EPhi(lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M)
        EPhiTPhi_reg = self.reg_EPhi(lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M, D)
        K_MM = self.kernel_gauss(U, lengthscale_trf, lengthscale_p_trf, sf_trf)
        XT_EPhi = X.T.dot(EPhi)
        opt_A_mean, cholSigInv, cholK_MM, InvK_MM = self.get_opt_A(sn_trf, EPhiTPhi, XT_EPhi, K_MM)
                
        LL = - 0.5 * (D * ((N - M) * T.log(sn_trf) + N * np.log(2 * np.pi) - T.sum(2 * T.log(T.diag(cholK_MM))) + T.sum(2 * T.log(T.diag(cholSigInv))) + (N * sf_trf - T.sum(T.diag(InvK_MM.dot(EPhiTPhi_reg))))/sn_trf) + T.sum(X ** 2)/sn_trf -  T.sum(opt_A_mean.T * XT_EPhi)/sn_trf)
        
        KL_X = - 0.5 * (T.log(2 * np.pi * SIGMA_trf) + 1).sum() + 0.5 * (np.log(2 * np.pi)) + 0.5 * (SIGMA_trf + MU**2).sum()
        
        return LL, KL_X
