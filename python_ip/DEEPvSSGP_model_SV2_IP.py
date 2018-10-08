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

class DEEPvSSGP:
    def __init__(self, Q, D, layers, order, D_cum_sum, N, M, non_rec):
        try:
            print('Trying to load model...')
            with open('model_SV2.save', 'rb') as file_handle:
                self.f, self.g = pickle.load(file_handle)
                print('Loaded!')
            return
        except:
            print('Failed. Creating a new model...')

        print('Setting up variables...')
        hyp, S, MU, SIGMA, U, b = T.dmatrices('hyp', 'S', 'MU', 'SIGMA', 'U', 'b')
        y, MEAN_MAP, sn, sf = T.dvectors('y','MEAN_MAP','sn','sf')
        w = T.dscalars('w')
        if Q > 1:
            X = T.dmatrix('X')
        else:
            X = T.dvector('X')     
        if layers > 1:
            MU, SIGMA = T.dmatrices('MU', 'SIGMA')
        else:
            MU, SIGMA = T.dvectors('MU', 'SIGMA')
                    
        SIGMA_trf = T.log(1+T.exp(SIGMA))**2       
        sf_trf, sn_trf, lengthscale_trf, lengthscale_p_trf  =  T.log(1 + T.exp(sf))**2, T.log(1 + T.exp(sn))**2, T.log(1 + T.exp(hyp[:,0])), T.log(1 + T.exp(hyp[:,1]))
        
        print('Setting up model...')
        LL, KL = self.get_model(w, lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, S, MU, SIGMA_trf, U, b, X, y, MEAN_MAP, Q, D, D_cum_sum, layers, order, non_rec, N, M)

        print('Compiling model...')
        
        inputs = {'X': X, 'MU': MU, 'SIGMA': SIGMA, 'S': S, 'U':  U, 'b':  b, 'hyp': hyp, 'y': y, 'MEAN_MAP': MEAN_MAP, 'sn': sn, 'sf': sf, 'w': w}
        z = 0.0 * sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = {'LL': LL, 'KL': KL}
        self.f = {fn: theano.function(list(inputs.values()), fv+z, name=fn, on_unused_input='ignore') for fn,fv in f.items()}       
        
        g = {'LL': LL, 'KL': KL}
        wrt = {'MU': MU, 'SIGMA': SIGMA, 'S':  S, 'U':  U, 'b':  b, 'hyp': hyp, 'MEAN_MAP': MEAN_MAP, 'sn': sn, 'sf': sf, 'w': w}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, on_unused_input='ignore') for gn,gv in g.items()} for vn, vv in wrt.items()}

        with open('model_SV2.save', 'wb') as file_handle:
            print('Saving model...')
            sys.setrecursionlimit(10000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def reg_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M, i, D, order, non_rec):
        
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
        def EPhiTPhi_loop_i0(loop, EPhiTPhi, non_rec, D, order, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp):
            loop = loop + 1
            D_n = (inv_B + inv_SIGMA_trf[loop,:][None,None,:])**-1 # M x M x D[i]
            if non_rec == 0:
            	d_n = D_n[:,:,D - order:D] * (inv_B_b_bold[:,:,D - order:D] + inv_SIGMA_trf_MU[loop,:][None,None,D - order:D]) # M x M x N x order
            	d_n = T.concatenate((MU[loop,:][0:D - order][None,None,:] + T.zeros_like(inv_B[:,:,0:D - order]),d_n), axis=2) # M x M x N x D[i]
            else:
            	d_n = MU[loop,:][None,None,:] + T.zeros_like(inv_B) # M x M x N x D[i]
            W = B + SIGMA_trf[loop,:][None,None,:]  # M x M x D[i]
            norm_EEPhiTPhi_U_W = (norm_EEPhiTPhi_U_temp/W**0.5).prod(2) # M x M  % here we put det(U), det(W), because of numeric issues (prod(2) is huge for huge input-dimensions) 
            Z_n_W =  T.exp(-0.5 * ((b_bold - MU[loop,:][None,None,:])**2 / W).sum(2)) # M x M   
            EPhiTPhi = EPhiTPhi + Z_n_W * norm_EEPhiTPhi_U_W * (T.exp(-0.5 * (MU_S_hat_minus**2 * D_n).sum(2)) * T.cos((MU_S_hat_minus * d_n).sum(2) + big_sum_minus) + T.exp(-0.5 * (MU_S_hat_plus**2 * D_n).sum(2)) * T.cos((MU_S_hat_plus * d_n).sum(2) + big_sum_plus)) # M x M
            return loop, EPhiTPhi
        
        def EPhiTPhi_loop_i(loop, EPhiTPhi, order, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp):
            loop = loop + 1
            D_n = (inv_B + inv_SIGMA_trf[loop,:][None,None,:])**-1 # M x M x D[i]
            d_n = D_n * (inv_B_b_bold + inv_SIGMA_trf_MU[loop,:][None,None,:]) # M x M x D[i]
            W = B + SIGMA_trf[loop,:][None,None,:]  # M x M x D[i]
            norm_EEPhiTPhi_U_W = (norm_EEPhiTPhi_U_temp/W**0.5).prod(2) # M x M  % here we put det(U), det(W), because of numeric issues (prod(2) is huge for huge input-dimensions) 
            Z_n_W =  T.exp(-0.5 * ((b_bold - MU[loop,:][None,None,:])**2 / W).sum(2)) # M x M   
            EPhiTPhi = EPhiTPhi + Z_n_W * norm_EEPhiTPhi_U_W * (T.exp(-0.5 * (MU_S_hat_minus**2 * D_n).sum(2)) * T.cos((MU_S_hat_minus * d_n).sum(2) + big_sum_minus) + T.exp(-0.5 * (MU_S_hat_plus**2 * D_n).sum(2)) * T.cos((MU_S_hat_plus * d_n).sum(2) + big_sum_plus))  # M x M
            return loop, EPhiTPhi
        
        if i == 0:
            result, _ = theano.scan(EPhiTPhi_loop_i0,
                                outputs_info = [loop, EPhiTPhi],
                                n_steps = N,
                                non_sequences = [non_rec, D, order, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp])
        else:
            result, _ = theano.scan(EPhiTPhi_loop_i,
                                outputs_info = [loop, EPhiTPhi],
                                n_steps = N,
                                non_sequences = [order, MU, SIGMA_trf, inv_SIGMA_trf, inv_SIGMA_trf_MU, inv_B, b_bold, inv_B_b_bold, B, MU_S_hat_minus, MU_S_hat_plus, big_sum_minus, big_sum_plus, norm_EEPhiTPhi_U_temp])

        EPhiTPhi_out = result[-1][-1] # M x M
        
        reg_EEPhiTPhi = (sf_trf**2/2) * Z_n_U_EEPhiTPhi * EPhiTPhi_out # M x M

        return reg_EEPhiTPhi

    def get_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, S, MU, SIGMA_trf, U, b, N, M): 
        
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

        S_hat = lengthscale_trf**-1 * S + 2 * np.pi * lengthscale_p_trf**-1 # M x D[i]        
        decay = T.exp(-0.5 * ((S_hat**2)[None,:, :] * SIGMA_trf[:, None, :]).sum(2)) # N x M x D[i] 
        cos_w = T.cos((S_hat[None,:, :] * (MU[:, None, :] - U[None, :, :])).sum(2) + b[None,:]) # N x M x D[i] 
        EPhi = (2 * sf_trf/M)**0.5 * decay * cos_w  # N x M

        S_hat_U_b =  -(S_hat * U).sum(1)[None,:] + b # M x M
        big_sum_minus = S_hat_U_b - S_hat_U_b.T # M x M
        big_sum_plus = S_hat_U_b + S_hat_U_b.T # M x M
        S_hat_minus = S_hat[None,:,:] - S_hat[:,None,:] # M x M x D[i]
        S_hat_plus = S_hat[None,:,:] + S_hat[:,None,:] # M x M x D[i] 

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
        
#        cos_2w =  T.cos(2 * (S_hat[None,:, :] * (MU[:, None, :] - U[None, :, :])).sum(2) + b[None,:]) # N x M
#        E_cos_sq = (sf_trf/M) * (1 + decay**4 * cos_2w) # N x M
#        sf_trf_vec = sf_trf * np.ones((N,1))

#        K_hat_reg = sf_trf_vec - E_cos_sq.sum(1) 

        return EPhi, EPhiTPhi
    
    def kernel_gauss(self, U, lengthscale_trf, lengthscale_p_trf, sf_trf):
        
        X_exp = U * lengthscale_trf**-1  # M x D[i] 
        X_cos = U * lengthscale_p_trf**-1 # M x D[i] 
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
    
    def update(self, layers, order, MU, SIGMA_trf, X, Q, D, D_cum_sum, N, non_rec):
        
        X_inputs = X;
        SIGMA_inputs = T.zeros_like(X);
                                   
        if layers > 1:
            if Q > 1:
                for i in range(0, order):
                    X_inputs = T.concatenate((X_inputs, MU[order - 1 - i:-i - 1,0][:,None]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs, SIGMA_trf[order - 1 - i:-i - 1,0][:,None]), axis=1)
            else:
                for i in range(0, order):
                    X_inputs = T.concatenate((X_inputs[:,None], MU[order - 1 - i:-i - 1,0][:,None]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs[:,None], SIGMA_trf[order - 1 - i:-i - 1,0][:,None]), axis=1)     
            if non_rec == 0:            
                for j in range(1, layers):
                    X_inputs = T.concatenate((X_inputs,MU[order:,j - 1][:,None],X_inputs[:,D_cum_sum[j] - order:-1]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order:,j - 1][:,None],SIGMA_inputs[:,D_cum_sum[j] - order:-1]), axis=1)
                    for i in range(0, order):
                        X_inputs = T.concatenate((X_inputs,MU[order - i - 1 :-i - 1,j][:,None]), axis=1)
                        SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order - i - 1:-i - 1,j][:,None]), axis=1)                       
                X_inputs = T.concatenate((X_inputs,MU[order:,layers - 1][:,None],X_inputs[:,D_cum_sum[layers] - order:-1]), axis=1)
                SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order:,layers - 1][:,None],SIGMA_inputs[:,D_cum_sum[layers] - order:-1]), axis=1)
            else:
                for j in range(1, layers):
                    X_inputs = T.concatenate((X_inputs,MU[order:,j - 1][:,None],X_inputs[:,D_cum_sum[j]:-1]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order:,j - 1][:,None],SIGMA_inputs[:,D_cum_sum[j]:-1]), axis=1)
                    for i in range(0, order):
                        X_inputs = T.concatenate((X_inputs,MU[order - i - 1 :-i - 1,j][:,None]), axis=1)
                        SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order - i - 1:-i - 1,j][:,None]), axis=1)                      
                X_inputs = T.concatenate((X_inputs[:,0:D_cum_sum[1]],X_inputs[:,D_cum_sum[1] + order:],MU[order:,layers - 1][:,None],X_inputs[:,D_cum_sum[layers]:-1]), axis=1)
                SIGMA_inputs = T.concatenate((SIGMA_inputs[:,0:D_cum_sum[1]],SIGMA_inputs[:,D_cum_sum[1] + order:],SIGMA_trf[order:,layers - 1][:,None],SIGMA_inputs[:,D_cum_sum[layers]:-1]), axis=1)
        else:
            if Q > 1:
                for i in range(0, order):
                    X_inputs = T.concatenate((X_inputs, MU[order - 1 - i:-i - 1][:,None]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs, SIGMA_trf[order - 1 - i:-i - 1][:,None]), axis=1)
            else:
                for i in range(0, order):
                    X_inputs = T.concatenate((X_inputs[:,None], MU[order - 1 - i:-i - 1][:,None]), axis=1)
                    SIGMA_inputs = T.concatenate((SIGMA_inputs[:,None], SIGMA_trf[order - 1 - i:-i - 1][:,None]), axis=1)   
            if non_rec == 0:                 
                X_inputs = T.concatenate((X_inputs,MU[order:][:,None],X_inputs[:,D_cum_sum[layers] - order:-1]), axis=1)
                SIGMA_inputs = T.concatenate((SIGMA_inputs,SIGMA_trf[order:][:,None],SIGMA_inputs[:,D_cum_sum[layers] - order:-1]), axis=1)
            else:                
                X_inputs = T.concatenate((X_inputs[:,0:D_cum_sum[1]],MU[order:][:,None],X_inputs[:,D_cum_sum[1]:-1]), axis=1)
                SIGMA_inputs = T.concatenate((SIGMA_inputs[:,0:D_cum_sum[layers]],SIGMA_trf[order:][:,None],SIGMA_inputs[:,D_cum_sum[1]:-1]), axis=1)
                               
        return X_inputs, SIGMA_inputs

    def get_model(self, w, lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, S, MU, SIGMA_trf, U, b, X, y, MEAN_MAP, Q, D, D_cum_sum, layers, order, non_rec, N, M):
        
        X_inputs,SIGMA_inputs = self.update(layers, order, MU, SIGMA_trf, X, Q, D, D_cum_sum, N, non_rec)
        LL = 0
        
        for i in range(0, layers + 1):
            EPhi, EPhiTPhi = self.get_EPhi(lengthscale_trf[D_cum_sum[i]:D_cum_sum[i + 1]], lengthscale_p_trf[D_cum_sum[i]:D_cum_sum[i + 1]], sf_trf[i], S[:,D_cum_sum[i]:D_cum_sum[i + 1]], X_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], SIGMA_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], U[:,D_cum_sum[i]:D_cum_sum[i + 1]], b[:,i], N, M)
            EPhiTPhi_reg = self.reg_EPhi(lengthscale_trf[D_cum_sum[i]:D_cum_sum[i + 1]], lengthscale_p_trf[D_cum_sum[i]:D_cum_sum[i + 1]], sf_trf[i], S[:,D_cum_sum[i]:D_cum_sum[i + 1]], X_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], SIGMA_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], U[:,D_cum_sum[i]:D_cum_sum[i + 1]], b[:,i], N, M, i, D[i], order, non_rec)
            if i == layers:
#                z = (1 + 2 * T.cos(w)**2) * y[order:]
                z =  y[order:]
                SIGMA_trf_LL = 0
            else:
                if layers > 1:
                    z = MU[order:,i] - X.dot(MEAN_MAP)
                    SIGMA_trf_LL = SIGMA_trf[order:,i]
                else:
                    z = MU[order:] - X.dot(MEAN_MAP)
                    SIGMA_trf_LL = SIGMA_trf[order:]
            K_MM = self.kernel_gauss(U[:,D_cum_sum[i]:D_cum_sum[i + 1]], lengthscale_trf[D_cum_sum[i]:D_cum_sum[i+1]], lengthscale_p_trf[D_cum_sum[i]:D_cum_sum[i+1]], sf_trf[i])

            zT_EPhi = z.T.dot(EPhi)
            opt_A_mean, cholSigInv, cholK_MM, InvK_MM = self.get_opt_A(sn_trf[i], EPhiTPhi, zT_EPhi, K_MM)
                    
            LL = LL - 0.5 * N* T.log(sn_trf[i]) - 0.5 * N * np.log(2 * np.pi)  + 0.5 * T.sum(2 * T.log(T.diag(cholK_MM))) - 0.5 * T.sum(2 * T.log(T.diag(cholSigInv))) - 0.5 * T.sum(SIGMA_trf_LL)/sn_trf[i] - 0.5 * T.sum(z ** 2)/sn_trf[i] + 0.5 * T.sum(opt_A_mean.T * zT_EPhi)/sn_trf[i]  - 0.5 * (N * sf_trf[i] - T.sum(T.diag(InvK_MM.dot(EPhiTPhi_reg))))/sn_trf[i]

        KL_X = - 0.5 * (T.log(2 * np.pi * SIGMA_trf) + 1).sum() + 0.5 * layers * order * (np.log(2 * np.pi)) + 0.5 * (SIGMA_trf[1:order,] + MU[1:order,]**2).sum()
        
        return LL, KL_X
