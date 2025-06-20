
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../..')

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices

import argparse

def LeastSquares(W, tau):
    return np.linalg.pinv(W) @ tau

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pick and Place Simulation')
    # 添加 hand_type 参数，默认值为 gx11
    parser.add_argument('--hand_type', type=str, default='gx11', help='Type of hand')
    # 添加 finger_idx 参数，默认值为 2
    parser.add_argument('--finger_idx', type=int, default=1, help='Index of finger')
    # 解析命令行参数
    args = parser.parse_args()

    hand_type = args.hand_type
    finger_idx = args.finger_idx

    rbt = get_robot(f'../../../data/model/gx11pm_finger{finger_idx+1}.pkl')

    joint_mask = np.zeros(11)
    if finger_idx == 0:
        joint_mask[:3] = 1
    elif finger_idx == 1:
        joint_mask[3:7] = 1
    elif finger_idx == 2:
        joint_mask[7:11] = 1
    joint_mask = joint_mask.astype(bool)
    num_joints = np.sum(joint_mask)
    print(f'num_joints: {num_joints}')
    # breakpoint()


    # load data
    data = np.load(f'data/traj/sensor_{hand_type}_{finger_idx}.npy').reshape((-1, 11, 3))
    print(data.shape)

    q = data[:, :, 0]
    qd = data[:, :, 1]
    tau = data[:, :, 2]

    qdd = (qd[1:]-qd[:-1])*240
    q = q[50+1:, joint_mask]
    qd = qd[50+1:, joint_mask]
    tau = tau[50+1:, joint_mask]
    qdd = qdd[50:, joint_mask]

    print(q.shape, qd.shape, qdd.shape, tau.shape)

    W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
    print(W.shape, omega.shape)

    beta_ols = LeastSquares(W, omega)
    pred_ols = (W @ beta_ols).reshape((-1, num_joints))
    print(beta_ols)

    np.save(f'data/params/{hand_type}_{finger_idx}.npy', beta_ols)
    
    