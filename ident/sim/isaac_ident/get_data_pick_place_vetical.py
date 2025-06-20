import sys, os
import numpy as np

sys.path.append('../../../')

from env.pick_place import PickPlace, orientation_error, control_ik
import torch
import scipy.signal as signal
# print(os.listdir('../../..'))


if __name__=="__main__":
    from scipy.spatial.transform import Rotation as R
    import argparse  # 导入 argparse 模块

    parser = argparse.ArgumentParser(description='Pick and Place Simulation')
    # 添加 hand_type 参数，默认值为 gx11
    parser.add_argument('--hand_type', type=str, default='gx11super', help='Type of hand')
    # 添加 finger_idx 参数，默认值为 2
    parser.add_argument('--finger_idx', type=int, default=0, help='Index of finger')
    # 解析命令行参数
    args = parser.parse_args()

    # 使用解析后的参数
    hand_type = args.hand_type
    finger_idx = args.finger_idx
    
    # 打印参数值
    print(f"Hand Type: {hand_type}")
    print(f"Finger Index: {finger_idx}")

    env = PickPlace(hand_type=hand_type, asset_root='../../../../isaac/assets/')
    env.pandas_initial_pos = torch.tensor([1.0102, -1.7628, -0.3847, -1.7226, 1.4329, 2.7606, -2.5097]).float()
    env.create_envs()
    
    # load traj
    # finger_idx = 2
    traj = np.load(f'../../../data/ident/pos_list_{finger_idx}_0.npy')*(np.pi/180)
    window_size = 100

    for j in range(11):
        traj[:, j] = signal.savgol_filter(traj[:, j], window_size, 3)
    print(traj.shape)
    

    # 1.0102, -1.7628, -0.3847, -1.7226, 1.4329, 2.7606, -2.5097, 1.2000, -0.0000, 0.0000,
    # 0.0000, -0.0000, 0.4000, -0.0000, 0.0000, -0.0000, 0.4000, -0.0000    
    action_initial = torch.tensor([1.0102, -1.7628, -0.3847, -1.7226, 1.4329, 2.7606, -2.5097,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(env.device).float()
    
    index = 0
    
    measure = []
    
    while not env.gym.query_viewer_has_closed(env.viewer):
        
        env.compute_observations()
        
        action_initial[7:] = torch.tensor(traj[index]).to(env.device).float()
        
        env.step(action_initial)
        
        # print('dof_pos:', env.dof_pos.shape)
        # print('dof_vel:', env.dof_vel.shape)
        # print('dof_force:', env.dof_force_tensor.shape)
        
        measure.append(np.array([env.dof_pos.cpu().numpy()[:, :, 0],
                        env.dof_vel.cpu().numpy()[:, :, 0],
                        env.dof_force_tensor.cpu().numpy()]))
        
    
        index += 1
        
        if index>=traj.shape[0]:
            break
    measure = np.array(measure)
    print(measure.shape)
    print('save the result')
    
    
    dir_save = '../../../data/ident_task/pick_place/traj'
    np.save(f'{dir_save}/measure_{hand_type}_{finger_idx}.npy', measure)
    
    
