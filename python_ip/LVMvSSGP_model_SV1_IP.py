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
            with open('model_SV1.save', 'rb') as file_handle:
                self.f, self.g = pickle.load(file_handle)
                print('Loaded!')
            return
        except:
            print('Failed. Creating a new model...')

        print('Setting up variables...')
        hyp, MU_S, SIGMA_S, MU, SIGMA, U, X = T.dmatrices('hyp', 'MU_S', 'SIGMA_S', 'MU', 'SIGMA', 'U', 'X')
        b = T.dvector('b')
        sn = T.dscalar('sn')
        sf = T.dscalar('sf') 

        SIGMA_trf, SIGMA_S_trf = T.log(1+T.exp(SIGMA))**2, T.log(1+T.exp(SIGMA_S))**2       
        sf_trf, sn_trf, lengthscale_trf, lengthscale_p_trf  =  T.log(1 + T.exp(sf))**2, T.log(1 + T.exp(sn))**2, T.log(1 + T.exp(hyp[:,0])), T.log(1 + T.exp(hyp[:,1]))
        
        print('Setting up model...')
        LL, KL = self.get_model(lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, MU_S, SIGMA_S_trf, MU, SIGMA_trf, U, b, X, Q, D, N, M)

        print('Compiling model...')
        
        inputs = {'X': X, 'MU': MU, 'SIGMA': SIGMA, 'MU_S': MU_S, 'SIGMA_S': SIGMA_S, 'U':  U, 'b':  b, 'hyp': hyp, 'sn': sn, 'sf': sf}
        z = 0.0 * sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = {'LL': LL, 'KL': KL}
        self.f = {fn: theano.function(list(inputs.values()), fv+z, name=fn, on_unused_input='ignore') for fn,fv in f.items()}       
        
        g = {'LL': LL, 'KL': KL}
        wrt = {'MU': MU, 'SIGMA': SIGMA, 'MU_S':  MU_S, 'SIGMA_S':  SIGMA_S, 'U':  U, 'b':  b, 'hyp': hyp, 'sn': sn, 'sf': sf}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, on_unused_input='ignore') for gn,gv in g.items()} for vn, vv in wrt.items()}

        with open('model_SV1.save', 'wb') as file_handle:
            print('Saving model...')
            sys.setrecursionlimit(100000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def reg_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, S, SIGMA_S, MU, SIGMA_trf, U, b, N, M, D):
        
#       lengthscale_trf # D[i]
#       lengthscale_p_trf # D[i]
#       sf_trf # 1
#       S # M x D[i]
#       MU # N x D[i]
#       SIGMA_trf # N x D[i]
#       U # M x D[i]
#       b # M
#       N # 1
#       M # 1
        
        b = T.zeros(T.shape(b))
        MU_S = T.zeros(T.shape(S))
        SIGMA_S_trf = T.ones(T.shape(S))
        
        inv_SIGMA_trf = SIGMA_trf**-1 # N x D[i]
        MU_S_hat = lengthscale_trf**-1 * MU_S + 2 * np.pi * lengthscale_p_trf**-1 # M x D[i]

        MU_S_hat_U_b =  -(MU_S_hat * U).sum(1)[None,:] + b # M x M
        big_sum_minus = MU_S_hat_U_b - MU_S_hat_U_b.T # M x M
        big_sum_plus = MU_S_hat_U_b + MU_S_hat_U_b.T # M x M
        MU_S_hat_minus = MU_S_hat[None,:,:] - MU_S_hat[:,None,:] # M x M x D[i]
        MU_S_hat_plus = MU_S_hat[None,:,:] + MU_S_hat[:,None,:] # M x M x D[i]
        
        u_EEPhiTPhi = (U[None,:,:] - U[:,None,:])**2 # M x M x D[i]
        b_bold_denomi = SIGMA_S_trf[None,:,:] + SIGMA_S_trf[:,None,:] # M x M x D[i]
        sum_SIGMA_S_U = SIGMA_S_trf * U # M x D[i]
        b_bold = (sum_SIGMA_S_U[None,:,:] + sum_SIGMA_S_U[:,None,:])/b_bold_denomi # M x M x D[i]
        B = (lengthscale_trf**2)[None,None,:]/b_bold_denomi # M x M x D[i]
        inv_B = 1/B  # M x M x D[i]
        U_EEPhiTPhi = (lengthscale_trf**2)[None,None,:]*(SIGMA_S_trf[None,:,:]**-1 + SIGMA_S_trf[:,None,:]**-1) # M x M x D[i]
        norm_EEPhiTPhi_U_temp = lengthscale_trf[None,None,:]**2/((SIGMA_S_trf[None,:,:] * SIGMA_S_trf[:,None,:]) * U_EEPhiTPhi)**0.5 # M x M x D[i]
        Z_n_U_EEPhiTPhi = np.exp(-0.5 * (u_EEPhiTPhi/U_EEPhiTPhi).sum(2))  # M x M        
        inv_B_b_bold = inv_B * b_bold # M x M x D[i]
        inv_SIGMA_trf_MU = inv_SIGMA_trf * MU # N x D[i]
 
        EPhiTPhi = np.zeros((M,M))
        loop = np.int64(-1) 

        def EPhiTPhi_loop_i(loop, EPhiTPhi, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp):
            loop = loop + 1
            D_n = (inv_B + inv_SIGMA_trf[loop,:][None,None,:])**-1 # M x M x D[i]
            d_n = D_n * (inv_B_b_bold + inv_SIGMA_trf_MU[loop,:][None,None,:]) # M x M x D[i]
            W = B + SIGMA_trf[loop,:][None,None,:]  # M x M x D[i]
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

    def get_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, MU_S, SIGMA_S_trf, MU, SIGMA_trf, U, b, N, M):
        
#       lengthscale_trf # D[i]
#       lengthscale_p_trf # D[i]
#       sf_trf # 1
#       S # M x D[i]
#       MU # N x D[i]
#       SIGMA_trf # N x D[i]
#       U # M x D[i]
#       b # M
#       N # 1
#       M # 1
        
        inv_SIGMA_trf = SIGMA_trf**-1 # N x D[i]
        SIGMA_S_trf_lengthscale_trf_quad = SIGMA_S_trf/(lengthscale_trf**2)[None,:] # M x D[i]
        SIGMA_S_trf_lengthscale_trf_quad_diag = SIGMA_S_trf/(lengthscale_trf**2/(2*2))[None,:] # M x D[i] diagonal
        lengthscale_trf_quad_SIGMA_S_trf = SIGMA_S_trf_lengthscale_trf_quad**-1 # M x D[i]
        lengthscale_trf_quad_SIGMA_S_trf_diag = SIGMA_S_trf_lengthscale_trf_quad_diag**-1 # M x D[i] diagonal
        cov_Z = lengthscale_trf_quad_SIGMA_S_trf[None,:,:] + SIGMA_trf[:,None,:] # N x M x D[i]
        C = (SIGMA_S_trf_lengthscale_trf_quad[None,:,:] + (SIGMA_trf**-1)[:,None,:])**-1 # N x M x D[i]
        cov_Z_diag = lengthscale_trf_quad_SIGMA_S_trf_diag[None,:,:] + SIGMA_trf[:,None,:] # N x M x D[i] for diagonal
        C_diag = (SIGMA_S_trf_lengthscale_trf_quad_diag[None,:,:] + (SIGMA_trf**-1)[:,None,:])**-1 # N x M x D[i] diagonal
        zeta = C*((SIGMA_S_trf_lengthscale_trf_quad * U)[None,:,:] + (MU/SIGMA_trf)[:,None,:]) - U[None,:,:] # N x M x D[i]
        zeta_diag = C_diag*((SIGMA_S_trf_lengthscale_trf_quad_diag * U)[None,:,:] + (MU/SIGMA_trf)[:,None,:]) - U[None,:,:] # N x M x D[i]
        norm_EEPhi = (lengthscale_trf_quad_SIGMA_S_trf[None,:,:]/cov_Z).prod(2) # N x M % we put cov_Z here, because of numeric issues (prod(2) is huge for huge input-dimensions)
        norm_EEPhi_diag = (lengthscale_trf_quad_SIGMA_S_trf_diag[None,:,:]/cov_Z_diag).prod(2) # N x M % we put cov_Z here, beause of numeric issues (prod(2) is huge for huge input-dimensions) diagonal
        Z = T.exp(-0.5 * ((U[None,:,:] - MU[:,None,:])**2/cov_Z).sum(2)) # N x M
        Z_diag = T.exp(-0.5 * ((U[None,:,:] - MU[:,None,:])**2/cov_Z_diag).sum(2)) # N x M diagonal
        MU_S_hat = lengthscale_trf**-1 * MU_S + 2 * np.pi * lengthscale_p_trf**-1 # M x D[i]
        
        decay = T.exp(-0.5 * ((MU_S_hat**2)[None,:, :] * C).sum(2)) # N x M 
        cos_w = T.cos((MU_S_hat[None,:, :] * zeta).sum(2) + b[None,:]) # N x M
        EEPhi = (2 * sf_trf * norm_EEPhi/M)**0.5 * Z * decay * cos_w  # N x M
        
        decay_diag = T.exp(-2 * ((MU_S_hat**2)[None,:, :] * C_diag).sum(2)) # N x M diagonal
        cos_w_diag = T.cos(2 * ((MU_S_hat[None,:, :] * zeta_diag).sum(2) + b[None,:])) # N x M diagonal
        EPhiTPhi_diag = (sf_trf)/M * (N + ((norm_EEPhi_diag)**0.5 * Z_diag * decay_diag * cos_w_diag).sum(0))  # N x M diagonal

        MU_S_hat_U_b =  -(MU_S_hat * U).sum(1)[None,:] + b # M x M
        big_sum_minus = MU_S_hat_U_b - MU_S_hat_U_b.T # M x M
        big_sum_plus = MU_S_hat_U_b + MU_S_hat_U_b.T # M x M
        MU_S_hat_minus = MU_S_hat[None,:,:] - MU_S_hat[:,None,:] # M x M x D[i]
        MU_S_hat_plus = MU_S_hat[None,:,:] + MU_S_hat[:,None,:] # M x M x D[i]
        
        u_EEPhiTPhi = (U[None,:,:] - U[:,None,:])**2 # M x M x D[i]
        b_bold_denomi = SIGMA_S_trf[None,:,:] + SIGMA_S_trf[:,None,:] # M x M x D[i]
        sum_SIGMA_S_U = SIGMA_S_trf * U # M x D[i]
        b_bold = (sum_SIGMA_S_U[None,:,:] + sum_SIGMA_S_U[:,None,:])/b_bold_denomi # M x M x D[i]
        B = (lengthscale_trf**2)[None,None,:]/b_bold_denomi # M x M x D[i]
        inv_B = 1/B  # M x M x D[i]
        U_EEPhiTPhi = (lengthscale_trf**2)[None,None,:]*(SIGMA_S_trf[None,:,:]**-1 + SIGMA_S_trf[:,None,:]**-1) # M x M x D[i]
        norm_EEPhiTPhi_U_temp = lengthscale_trf[None,None,:]**2/((SIGMA_S_trf[None,:,:] * SIGMA_S_trf[:,None,:]) * U_EEPhiTPhi)**0.5 # M x M x D[i]
        Z_n_U_EEPhiTPhi = np.exp(-0.5 * (u_EEPhiTPhi/U_EEPhiTPhi).sum(2))  # M x M        
        inv_B_b_bold = inv_B * b_bold # M x M x D[i]
        inv_SIGMA_trf_MU = inv_SIGMA_trf * MU # N x D[i]
 
        EPhiTPhi = np.zeros((M,M))
        loop = np.int64(-1) 

        def EPhiTPhi_loop_i(loop, EPhiTPhi, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp):
            loop = loop + 1
            D_n = (inv_B + inv_SIGMA_trf[loop,:][None,None,:])**-1 # M x M x D[i]
            d_n = D_n * (inv_B_b_bold + inv_SIGMA_trf_MU[loop,:][None,None,:]) # M x M x D[i]
            W = B + SIGMA_trf[loop,:][None,None,:]  # M x M x D[i]
            norm_EEPhiTPhi_U_W = (norm_EEPhiTPhi_U_temp/W**0.5).prod(2) # M x M  % here we put det(U), det(W), because of numeric issues (prod(2) is huge for huge input-dimensions) 
            Z_n_W =  T.exp(-0.5 * ((b_bold - MU[loop,:][None,None,:])**2 / W).sum(2)) # M x M   
            EPhiTPhi = EPhiTPhi + Z_n_W * norm_EEPhiTPhi_U_W * (T.exp(-0.5 * (MU_S_hat_minus**2 * D_n).sum(2)) * T.cos((MU_S_hat_minus * d_n).sum(2) + big_sum_minus) + T.exp(-0.5 * (MU_S_hat_plus**2 * D_n).sum(2)) * T.cos((MU_S_hat_plus * d_n).sum(2) + big_sum_plus))  # M x M
            return loop, EPhiTPhi
        
        result, _ = theano.scan(EPhiTPhi_loop_i,
                            outputs_info = [loop, EPhiTPhi],
                            n_steps = N,
                            non_sequences = [MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp])

        EPhiTPhi_out = result[-1][-1] # M x M
        
        EEPhiTPhi = (sf_trf/M) * Z_n_U_EEPhiTPhi * EPhiTPhi_out # M x M
                    
        EEPhiTPhi = EEPhiTPhi - T.diag(T.diag(EEPhiTPhi)) + T.diag(EPhiTPhi_diag) # M x M

        return EEPhi, EEPhiTPhi
    
    def kernel_gauss(self, U, lengthscale_trf, lengthscale_p_trf, sf_trf):
        
        X_exp = U * lengthscale_trf**-1  # M x Q 
        X_cos = U * lengthscale_p_trf**-1 # M x Q 
        X_exp_product = (X_exp**2).sum(1) # M
        
        dquad = -2 * X_exp.dot(X_exp.T) + (X_exp_product.T)[None,:] + X_exp_product[:,None] # M x M
        dsub = (X_cos[None,:,:] - X_cos[:,None,:]).sum(2) # M x M
        K = sf_trf * T.exp(-1/2 * dquad) * T.cos(2 * np.pi * dsub) # M x M
               
        return K
    

    def get_opt_A(self, sn_trf, EPhiTPhi, XT_EPhi, K_MM):
        cholSigInv = sT.cholesky(EPhiTPhi + sn_trf * K_MM + 1e-6 * T.identity_like(K_MM))
        cholK_MM = sT.cholesky(K_MM + 1e-6 * T.identity_like(K_MM))
        invCholSigInv = sT.matrix_inverse(cholSigInv)
        invCholK_MM = sT.matrix_inverse(cholK_MM)
        InvSig = invCholSigInv.T.dot(invCholSigInv)
        InvK_MM = invCholK_MM.T.dot(invCholK_MM)
        Sig_EPhiT_X = InvSig.dot(XT_EPhi.T)
        return Sig_EPhiT_X, cholSigInv, cholK_MM, InvK_MM

    def get_model(self, lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, MU_S, SIGMA_S, MU, SIGMA_trf, U, b, X, Q, D, N, M):
                
        EPhi, EPhiTPhi = self.get_EPhi(lengthscale_trf, lengthscale_p_trf, sf_trf, MU_S, SIGMA_S, MU, SIGMA_trf, U, b, N, M)
        EPhiTPhi_reg = self.reg_EPhi(lengthscale_trf, lengthscale_p_trf, sf_trf, MU_S, SIGMA_S, MU, SIGMA_trf, U, b, N, M, D)
        K_MM = self.kernel_gauss(U, lengthscale_trf, lengthscale_p_trf, sf_trf)
        XT_EPhi = X.T.dot(EPhi)
        opt_A_mean, cholSigInv, cholK_MM, InvK_MM = self.get_opt_A(sn_trf, EPhiTPhi, XT_EPhi, K_MM)
                
        LL = - 0.5 * (D * ((N - M) * T.log(sn_trf) + N * np.log(2 * np.pi) - T.sum(2 * T.log(T.diag(cholK_MM))) + T.sum(2 * T.log(T.diag(cholSigInv))) + (N * sf_trf - T.sum(T.diag(InvK_MM.dot(EPhiTPhi_reg))))/sn_trf) + T.sum(X ** 2)/sn_trf -  T.sum(opt_A_mean.T * XT_EPhi)/sn_trf)
        
        KL_S = 0.5 * (SIGMA_S + MU_S**2 - T.log(SIGMA_S) - 1).sum()
        KL_X = - 0.5 * (T.log(2 * np.pi * SIGMA_trf) + 1).sum() + 0.5 * (np.log(2 * np.pi)) + 0.5 * (SIGMA_trf + MU**2).sum()
        KL = KL_S + KL_X

        return LL, KL
