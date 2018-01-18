# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle

print('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.exception_verbosity = 'high'
theano.config.reoptimize_unpickled_function = True

class DEEPvSSGP:
    def __init__(self, Q, D, layers, order, D_cum_sum, N, M, non_rec):
        try:
            print('Trying to load model...')
            with open('model_SV1.save', 'rb') as file_handle:
                self.f, self.g = pickle.load(file_handle)
                print('Loaded!')
            return
        except:
            print('Failed. Creating a new model...')

        print('Setting up variables...')
        hyp, SIGMA_S, U, b, MU_S = T.dmatrices('hyp', 'SIGMA_S', 'U', 'b','MU_S')
        y, MEAN_MAP, sn, sf = T.dvectors('y','MEAN_MAP','sn','sf')
        if Q > 1:
            X = T.dmatrix('X')
        else:
            X = T.dvector('X')     
        if layers > 1:
            MU, SIGMA = T.dmatrices('MU', 'SIGMA')
        else:
            MU, SIGMA = T.dvectors('MU', 'SIGMA')        
        
#        MU_S = T.stack([T.squeeze(MU_S)]*M, axis=1)
        SIGMA_trf, SIGMA_S_trf = T.log(1+T.exp(SIGMA))**2, T.log(1+T.exp(SIGMA_S))**2       
        sf_trf, sn_trf, lengthscale_trf, lengthscale_p_trf  =  T.log(1 + T.exp(sf))**2, T.log(1 + T.exp(sn))**2, T.log(1 + T.exp(hyp[:,0])), T.log(1 + T.exp(hyp[:,1]))
        
        print('Setting up model...')
        LL, KL = self.get_model(lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, MU_S, SIGMA_S_trf, MU, SIGMA_trf, U, b, X, y, MEAN_MAP, Q, D, D_cum_sum, layers, order, non_rec, N, M)

        print('Compiling model...')
        
        inputs = {'X': X, 'MU': MU, 'SIGMA': SIGMA, 'MU_S': MU_S, 'SIGMA_S': SIGMA_S, 'U':  U, 'b':  b, 'hyp': hyp, 'y': y, 'MEAN_MAP': MEAN_MAP, 'sn': sn, 'sf': sf}
        z = 0.0 * sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = {'LL': LL, 'KL': KL}
        self.f = {fn: theano.function(list(inputs.values()), fv+z, name=fn, on_unused_input='ignore') for fn,fv in f.items()}  
                  
        g = {'LL': LL, 'KL': KL}
        wrt = {'MU': MU, 'SIGMA': SIGMA, 'MU_S': MU_S, 'SIGMA_S': SIGMA_S, 'U':  U, 'b':  b, 'hyp': hyp, 'MEAN_MAP': MEAN_MAP,  'sn': sn, 'sf': sf}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, on_unused_input='ignore') for gn,gv in g.items()} for vn, vv in wrt.items()}


        with open('model_SV1.save', 'wb') as file_handle:
            print('Saving model...')
            sys.setrecursionlimit(10000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_EPhi(self, lengthscale_trf, lengthscale_p_trf, sf_trf, MU_S, SIGMA_S_trf, MU, SIGMA_trf, U, b, N, M, i, D, order, non_rec): 
        
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
        if i == 0:
            if non_rec == 0:
            	zeta_temp = C[:,:,D - order:D]*((SIGMA_S_trf_lengthscale_trf_quad * U)[None,:,D - order:D] + (MU/SIGMA_trf)[:,None,D - order:D]) # N x M x order
            	zeta_temp = T.concatenate((MU[:,0:D - order][:,None,:] + np.zeros((N,M,D - order)),zeta_temp), axis=2) # N x M x D[i]
            	zeta = zeta_temp - U[None,:,:] # N x M x D[i]
            	zeta_temp_diag = C_diag[:,:,D - order:D]*((SIGMA_S_trf_lengthscale_trf_quad_diag * U)[None,:,D - order:D] + (MU/SIGMA_trf)[:,None,D - order:D]) # N x M x order
            	zeta_temp_diag = T.concatenate((MU[:,0:D - order][:,None,:] + np.zeros((N,M,D - order)),zeta_temp_diag), axis=2) # N x M x D[i]
            	zeta_diag = zeta_temp_diag - U[None,:,:]# N x M x D[i]
            else:
            	zeta = MU[:,None,:] - U[None,:,:] # N x M x D[i]
            	zeta_diag = zeta # N x M x D[i]
        else:
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
        
        EEPhiTPhi = (sf_trf/M) * Z_n_U_EEPhiTPhi * EPhiTPhi_out # M x M
                    
        EEPhiTPhi = EEPhiTPhi - T.diag(T.diag(EEPhiTPhi)) + T.diag(EPhiTPhi_diag) # M x M

        return EEPhi, EEPhiTPhi

    def get_opt_A(self, sn_trf, EEPhiTPhi, XT_EEPhi):
        SigInv = EEPhiTPhi + (sn_trf + 1e-6) * T.identity_like(EEPhiTPhi)
        cholSigInv = sT.cholesky(SigInv)
        invCholSigInv = sT.matrix_inverse(cholSigInv)
        InvSig = invCholSigInv.T.dot(invCholSigInv)
        Sig_EEPhiT_X = InvSig.dot(XT_EEPhi.T)
        return Sig_EEPhiT_X, cholSigInv
    
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

    def get_model(self, lengthscale_trf, lengthscale_p_trf, sn_trf, sf_trf, MU_S, SIGMA_S_trf, MU, SIGMA_trf, U, b, X, y, MEAN_MAP, Q, D, D_cum_sum, layers, order, non_rec, N, M):
        
        X_inputs,SIGMA_inputs = self.update(layers, order, MU, SIGMA_trf, X, Q, D, D_cum_sum, N, non_rec)
        LL = 0
        
        for i in range(0, layers + 1):
            EEPhi, EEPhiTPhi = self.get_EPhi(lengthscale_trf[D_cum_sum[i]:D_cum_sum[i + 1]], lengthscale_p_trf[D_cum_sum[i]:D_cum_sum[i + 1]], sf_trf[i], MU_S[:,D_cum_sum[i]:D_cum_sum[i + 1]], SIGMA_S_trf[:,D_cum_sum[i]:D_cum_sum[i + 1]], X_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], SIGMA_inputs[:,D_cum_sum[i]:D_cum_sum[i + 1]], U[:,D_cum_sum[i]:D_cum_sum[i + 1]], b[:,i], N, M, i, D[i], order, non_rec)
            if i == layers:
                z = y[order:]
                SIGMA_trf_LL = 0
            else:
                if layers > 1:
                    z = MU[order:,i] - X.dot(MEAN_MAP)
                    SIGMA_trf_LL = SIGMA_trf[order:,i]
                else:
                    z = MU[order:] - X.dot(MEAN_MAP)
                    SIGMA_trf_LL = SIGMA_trf[order:]
            zT_EEPhi = z.T.dot(EEPhi)
            opt_A_mean, cholSigInv = self.get_opt_A(sn_trf[i], EEPhiTPhi, zT_EEPhi)
            LL = LL - 0.5 * (N - M) * T.log(sn_trf[i]) - 0.5 * N * np.log(2 * np.pi) - 0.5 * T.sum(2 * T.log(T.diag(cholSigInv))) - 0.5 * T.sum(SIGMA_trf_LL)/sn_trf[i] - 0.5 * T.sum(z ** 2)/sn_trf[i] + 0.5 * T.sum(opt_A_mean.T * zT_EEPhi)/sn_trf[i]    

        KL_S = 0.5 * (SIGMA_S_trf + MU_S**2 - T.log(SIGMA_S_trf) - 1).sum()
        KL_X = -0.5 * (T.log(2 * np.pi * SIGMA_trf) + 1).sum() + 0.5 * layers * order * (np.log(2 * np.pi)) + 0.5 * (SIGMA_trf[1:order,] + MU[1:order,]**2).sum()
        KL = KL_S + KL_X
        
        return LL, KL
