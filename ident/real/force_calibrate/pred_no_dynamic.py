import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.append('../../..')

from ident.sim.env import GX11PMEnv

from utils.ForceEstimation import HandForce

if __name__ == '__main__':
    np.random.seed(42)

    hand_force = HandForce(hand_urdf_path='/home/gx4070/data/lx/DynaForce/data/GX11promax/urdf/GX11promax.urdf')
    hand_force.load_default_params(dir='../../../data')

    data_dir = 'data/thumb_vertical_long' # 'data/thumb_horizontal' 'data/thumb_vertical'
    joint_mask = np.zeros(11)
    joint_mask[:3] = 1
    joint_mask = joint_mask.astype(bool)

    default_joint = [np.pi/4, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0]

    env_pybullet = GX11PMEnv(hand_urdf_path='/home/gx4070/data/lx/DynaForce/data/GX11promax/urdf/GX11promax.urdf', 
                             headless=True)
    jacobian_np = env_pybullet.get_jaccobian(joint_pos=default_joint, link_idx=3)
    jacobian_np[np.abs(jacobian_np)<0.01] = 0

    # load the data
    pos = np.load(f'{data_dir}/pos_list.npy')[:1500]
    torque = np.load(f'{data_dir}/torque_list.npy')[:1500]
    time_list = np.load(f'{data_dir}/t_list.npy')[:1500]
    force = np.load(f'{data_dir}/force_list.npy')[:1500]
    
    # breakpoint()

    torque[torque>32767] = torque[torque>32767]- 65536
    # print(pos.shape, torque.shape, torque.shape, force.shape)

    torque_norm = 1000

    # breakpoint()
    res_tau = torque[:, :3]/torque_norm
    force_pred_no_ident = np.linalg.pinv(jacobian_np.round(2).T)[:, joint_mask]@res_tau.T
    print(force_pred_no_ident.shape)
    np.save(f'{data_dir}/force_pred_no_ident.npy', force_pred_no_ident.T)
    # breakpoint()
    force_pred = np.linalg.norm(force_pred_no_ident, axis=0)
    print(force_pred.shape)
