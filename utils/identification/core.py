import numpy as np

import sys


sys.path.append('..')
sys.path.append('../../..')


from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices, get_momentum_regr_matrices
from utils.identification.gene_robot import get_robot

from utils.identification.algo.ols import LeastSquares
from utils.identification.algo.wls import weightedLeastSquares
from utils.identification.algo.irls import iterativeReweightedLeastSquares
from utils.identification.algo.ransac import inertial_ransac
from utils.identification.algo.lmi import FBPE

class IdentFingaer:
    def __init__(self, rbt_path, joint_mask=None, K=None):
        self.rbt = get_robot(rbt_path)
        if joint_mask is None:
            self.joint_mask = np.ones(self.rbt.dof).astype(bool)
        else:
            self.joint_mask = joint_mask.astype(bool)
            
        self.momentum_r = np.zeros((1, self.rbt.dof))
        
        self.dp_Integ = np.zeros((1, self.rbt.dof))
        
        if K is None:
            self.K = np.ones((1, self.rbt.dof))*5
            
        self.last_M = np.zeros((1, self.rbt.dof))
        # self.last_M = None
        
    def set_K(self, K):
        self.K = K
        
    
    def ident_params(self, q, qd, qdd, tau, algo='ols'):
        q = q[:, self.joint_mask]
        qd = qd[:, self.joint_mask]
        qdd = qdd[:, self.joint_mask]
        tau = tau[:, self.joint_mask]

        W, omega, Q1, R1, rho1 = gen_regr_matrices(self.rbt, q, qd, qdd, tau)
        # breakpoint()
        if algo == 'ols':    
            beta = LeastSquares(W, omega)
            self.beta = beta
        elif algo == 'wls':
            beta, _, _ = weightedLeastSquares(W, omega, nDof=self.rbt.dof)
            self.beta = beta
        elif algo == 'irls':
            beta_irls, weight_vector = iterativeReweightedLeastSquares(W, omega, nDof=self.rbt.dof,
	                                    thresholdOmega=1e-5, thresholdWeight=0.001)
            self.beta = beta_irls
        elif algo=='ransac':
            beta_best = inertial_ransac(W, omega, tau, self.rbt.dof, self.rbt.dyn.n_base,
                                        n_data=100, n_p=30, nI=100, max_distance=0.3)
            self.beta = beta_best
        elif algo == 'lmi':
            beta_init, beta_prime, beta_star = FBPE(W, omega, R1, rho1, self.rbt)
            self.beta = beta_star
        elif algo == 'lmi_ransac':
            beta_best = inertial_ransac(W, omega, tau, self.rbt.dof, self.rbt.dyn.n_base,
                            n_data=100, n_p=30, nI=100, max_distance=0.3)
            beta_init, beta_prime, beta_star = FBPE(W, omega, R1, rho1, self.rbt, beta_init=beta_best)
            self.beta = beta_star
        else:
            raise ValueError('algo not supported')
    
    def set_params(self, beta):
        self.beta = beta
    
    def load_params(self, dyn_path):
        pass
    
    def pred_torque(self, q, qd, qdd):
        q = q[:, self.joint_mask]
        qd = qd[:, self.joint_mask]
        qdd = qdd[:, self.joint_mask]
        tau_fake = q*0
        num_c = np.sum(self.joint_mask)
        W, omega, Q1, R1, rho1 = gen_regr_matrices(self.rbt, q, qd, qdd, tau_fake)

        return (W @ self.beta).reshape((-1, num_c))

    def pred_torque_momentum(self, q, qd, qdd, tau_measure):
        q = q[:, self.joint_mask]
        qd = qd[:, self.joint_mask]
        qdd = qdd[:, self.joint_mask]

        
        num_c = np.sum(self.joint_mask)

        
        
        # Y1 = M(q) dq + C(q, dq) dq + G(q)
        # Y2 = C(q, dq) dq + G(q)
        # Y3 = G(q)
        # M(q) dq = Y1-Y2
        # C(q, dq) dq = Y2-Y3
        zero_pad = np.zeros_like(qd)
        
        # # create a daig matrix
        diag_matrix = np.eye(num_c)
        zero_pad_matrix = np.zeros_like(diag_matrix) 
        q_tile = np.tile(q, (num_c, 1))
        # breakpoint()
        
        q_extent = np.concatenate([q,         q,        q,        q_tile,         q_tile, ], axis=0)
        qd_extent = np.concatenate([zero_pad, qd,       zero_pad, zero_pad_matrix, zero_pad_matrix], axis=0)
        qdd_extent = np.concatenate([qd,      zero_pad, zero_pad, diag_matrix,    zero_pad_matrix], axis=0)
        tau_fake = q_extent*0

        W_extent, omega_extent, Q1_extent, R1_extent, rho1_extent = gen_regr_matrices(self.rbt, q_extent, qd_extent, qdd_extent, tau_fake)

        Y_extent = (W_extent @ self.beta).reshape((-1, num_c))
        
        Y1 = Y_extent[0]
        Y2 = Y_extent[1]
        Y3 = Y_extent[2]
        Y4 = Y_extent[3:3+num_c]- Y_extent[3+num_c:3+num_c*2]

        dt = 0.1
        
        # tau_pred = np.asarray(Y4)
        Mdq = np.asarray(Y1-Y3)
        G_plus_Cdq = np.asarray(Y2)
        M = np.asarray(Y4.T)

        dM = (M-self.last_M)/dt
        # print()
        dMdq = np.dot(dM, qd.T).T

        eta = G_plus_Cdq - dMdq
        
        self.dp_Integ += (tau_measure - eta + self.momentum_r)*dt
        
        self.momentum_r = self.K * np.array(Mdq - self.dp_Integ)
        
        self.last_M = M*1.0
        
        return self.momentum_r


def process_data(q, torque, t_list, urdf_sign=None, torque_norm=1000):
    if urdf_sign is None:
        urdf_sign = np.array([-1, -1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1])[None]
    q = q*(np.pi/180)*urdf_sign
    delta_t = (t_list[1:]-t_list[:-1]).mean()
    print('delta_t: ', delta_t)
    qd = (q[1:]-q[:-1])/delta_t
    qdd = (qd[1:]-qd[:-1])/delta_t
    torque[torque>32767] = torque[torque>32767]- 65536
    torque = torque*urdf_sign/torque_norm
    
    # breakpoint()
    q = q[2:]
    qd = qd[1:]
    qdd = qdd[:]
    torque = torque[2:]
    
    return q, qd, qdd, torque
