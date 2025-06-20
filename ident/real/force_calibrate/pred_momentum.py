import matplotlib.pyplot as plt
import numpy as np


import sys
sys.path.append('../../..')

from ident.sim.env import GX11PMEnv


from utils.ForceEstimation import HandForce

if __name__ == '__main__':
    np.random.seed(42)


    import argparse

    parser = argparse.ArgumentParser(description='Predict momentum with force calibration.')
    parser.add_argument('--K', type=float, default=10)
    args = parser.parse_args()
    
    K = args.K
    print(K)

    

    hand_force = HandForce(hand_urdf_path='/home/gx4070/data/lx/DynaForce/data/GX11promax/urdf/GX11promax.urdf')
    hand_force.load_default_params(dir='../../../data')

    data_dir = 'data/thumb_vertical_long' # 'data/thumb_horizontal' 'data/thumb_vertical'
    joint_mask = np.zeros(11)
    joint_mask[:3] = 1
    joint_mask = joint_mask.astype(bool)

    default_joint = [np.pi/4, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0]


    # load the data
    pos = np.load(f'{data_dir}/pos_list.npy')[1000:1500]
    torque = np.load(f'{data_dir}/torque_list.npy')[1000:1500]
    time_list = np.load(f'{data_dir}/t_list.npy')[1000:1500]
    force = np.load(f'{data_dir}/force_list.npy')[1000:1500]

    torque[torque>32767] = torque[torque>32767]- 65536
    print(pos.shape, torque.shape, torque.shape, force.shape)

    torque_norm = 1000

    last_acc = np.zeros(11)
    last_vel = np.zeros(11)
    last_pos = np.zeros(11)

    ema_factor = [0.8, 0.8, 0.8]
    force_fingers = [
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
    ]
    
    
    for finger_idx, finger in enumerate(hand_force.finger_list):
        finger.set_K(np.ones((1, finger.rbt.dof))*K)
        
        
    
    last_time = 0.0
    
    force_pred_list = []
    force_pred_raw_list = []
    for t in range(pos.shape[0]-2):
        if t%100==0: print(t)
        
        cur_t = time_list[t]
        cur = torque[t]*hand_force.urdf_sign[0]/torque_norm

        cur_pos = pos[t].copy()*(np.pi/180)*hand_force.urdf_sign[0]
        cur_vel = (cur_pos - last_pos) / (cur_t - last_time)
        cur_acc = (cur_vel - last_vel) / (cur_t - last_time)
        
        if t>=10:
            force_pred = hand_force.estimate_force_momentum(cur_pos, cur_vel, cur_acc, cur, default_joint)

            force_pred_list.append(force_fingers[0].copy())
            force_pred_raw_list.append(force_pred[0].copy())

    
        last_pos = cur_pos.copy()
        last_vel = cur_vel
        last_acc = cur_acc
        last_time = cur_t

    
    force_pred_list = np.array(force_pred_list)
    force_norm = np.linalg.norm(force_pred_list, axis=(1, 2))
    # plt.figure(figsize=(10, 10))
    # plt.scatter(np.abs(force[10:]), np.abs(force_norm))

    np.save(f'{data_dir}/force_pred_list_momentum_{K}.npy', force_pred_list)
    
    # force_pred_raw_list
    np.save(f'{data_dir}/force_pred_raw_list_momentum_{K}.npy', force_pred_raw_list)


