
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../..')

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices

import scipy.signal as signal

def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau


if __name__ == '__main__':

    joint_idx = 2
    exp_idx = 0
    
    if joint_idx == 0:
        c_begin = 0
        c_end = 3
        num_c = 3
    elif joint_idx == 1:
        c_begin = 3
        c_end = 7
        num_c = 4
    elif joint_idx == 2:
        c_begin = 7
        c_end = 11
        num_c = 4

    rbt = get_robot(f'../../data/model/gx11pm_finger{joint_idx+1}.pkl')
    t_list = np.load(f'../../data/ident/t_list_{joint_idx}_{exp_idx}.npy')
    urdf_sign = np.array([-1, -1, 1, 
                          1, 1, 1, 1, 
                          1, 1, 1, 1])[None]

    delta_t = (t_list[1:]-t_list[:-1]).mean()
    print(1/delta_t)

    # load data
    q = np.load(f'../../data/ident/pos_list_{joint_idx}_{exp_idx}.npy')*(np.pi/180)*urdf_sign
    print(q.shape)
    qd = (q[1:]-q[:-1])#/delta_t
    qdd = (qd[1:]-qd[:-1])#/delta_t
    torque = np.load(f'../../data/ident/torque_list_{joint_idx}_{exp_idx}.npy')
    
    torque[torque>32767] = torque[torque>32767]- 65536
    torque = torque*urdf_sign/100

    q = q[50:450, c_begin:c_end] 
    qd = qd[50:450, c_begin:c_end]
    qdd = qdd[50:450, c_begin:c_end]
    tau = torque[50:450, c_begin:c_end]
    print(q.shape, qd.shape, qdd.shape, tau.shape)

    W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
    print(W.shape, omega.shape)

    beta_ols = LeastSquares(W, omega)
    pred_ols = (W @ beta_ols).reshape((-1, num_c))

    error = pred_ols - tau
    print(f'error: {np.abs(error).mean()}')
    print(pred_ols.shape, tau.shape)
    # plot in 3 subplots
    for i in range(num_c):
        plt.subplot(num_c, 1, i+1)
        plt.plot(pred_ols[:, i], label='pred')
        plt.plot(tau[:, i], label='gt')
        

    plt.legend()
    plt.show()

    print(pred_ols.shape)

    for i in range(2, 2):
        plt.subplot(2, 2, 1+i)
        plt.plot(pred_ols[:, i], label='pred')
        plt.plot(tau[:, i], label=f'gt')
        plt.title(f'J {i}')
    plt.show()



