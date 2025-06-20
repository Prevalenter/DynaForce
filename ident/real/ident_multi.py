
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../..')


from utils.identification.core import IdentFingaer, process_data

from utils.identification.metric import mse, relative_error

from sklearn.metrics import mean_absolute_percentage_error

import scipy.signal as signal
import time


if __name__ == '__main__':
    # fix the seed of numpy and random
    np.random.seed(42)
    
    
    # joint_idx, exp_idx, algo = 3, 4, 6
    # exp_all = [0, 1, 2, 3, 4]
    
    rst_mse = np.zeros((3, 3, 6))
    rst_rel_err = np.zeros((3, 3, 6))
    
    # = ['ols', 'wls', 'irls', 'ransac', 'lmi', 'lmi_ransac']
    # algo_dict = {
    #     'ols': 0,
    #     'wls': 1,
    #     'irls': 2,
    #     'ransac': 3,
    #     'lmi': 4,
    #     # 'lmi_ransac': 5
    # }
    algo_dict = {
        'ransac': 0,
        'wls': 1,
        'irls': 2,
        'lmi': 3,
    }
    algo_list = list(algo_dict.keys())
    
    for joint_idx in [0, 1, 2]:
        print(f'joint_idx: {joint_idx}')

        exp_idx = 2
        
        joint_mask = np.zeros(11)
        if joint_idx == 0:
            joint_mask[:3] = 1
        elif joint_idx == 1:
            joint_mask[3:7] = 1
        elif joint_idx == 2:
            joint_mask[7:11] = 1
        # joint_mask[:3] = 1
        joint_mask = joint_mask.astype(bool)
        
        finger = IdentFingaer(f'../../data/model/gx11pm_finger{joint_idx+1}.pkl', joint_mask=joint_mask)
       
        
        q_raw = np.load(f'../../data/ident/pos_list_{joint_idx}_{exp_idx}.npy')
        torque_raw = np.load(f'../../data/ident/torque_list_{joint_idx}_{exp_idx}.npy')
        t_list_raw = np.load(f'../../data/ident/t_list_{joint_idx}_{exp_idx}.npy')


        for algo in algo_list:
            print('-'*40)
            print(f'algo: {algo}')
            mse_list = []
            rel_err_list = []
            for exp_save_idx, exp_idx in enumerate([0, 1, 3]):

                # get beta
                q, qd, qdd, torque = process_data(q_raw.copy(), torque_raw.copy(), t_list_raw.copy())
                finger.ident_params(q, qd, qdd, torque, algo=algo)
                np.save(f'../../data/ident/beta/{algo}_beta_finger_{joint_idx}.npy', finger.beta)
                
                q = np.load(f'../../data/ident/pos_list_{joint_idx}_{exp_idx}.npy')
                torque = np.load(f'../../data/ident/torque_list_{joint_idx}_{exp_idx}.npy')
                t_list = np.load(f'../../data/ident/t_list_{joint_idx}_{exp_idx}.npy')

                q, qd, qdd, torque = process_data(q, torque, t_list)

                tau_pred = finger.pred_torque(q, qd, qdd)
                # print(tau_pred.shape, torque.shape,torque[:, joint_mask].shape)
                mse_val = mse(tau_pred*10, torque[:, joint_mask]*10)
                rel_err = relative_error(tau_pred, torque[:, joint_mask])
                # rel_err = mean_absolute_percentage_error(np.asarray(torque[:, joint_mask]),
                #                                          np.asarray(tau_pred)+100
                #                                          )
                
                
                mse_list.append(mse_val)
                rel_err_list.append(rel_err)
                
                rst_mse[joint_idx, exp_save_idx, algo_dict[algo]] = mse_val
                rst_rel_err[joint_idx, exp_save_idx, algo_dict[algo]] = rel_err
                
            mse_val_mean = np.mean(mse_list)
            rel_err_mean = np.mean(rel_err_list)

            
            print(f'mse: {mse_val_mean}')
            print(f'rel_err: {rel_err_mean}')
            
            # print(tau_pred.shape, torque.shape)
        
        # for i in range(3):
        #     plt.subplot(3, 1, i+1)
        #     plt.plot(tau_pred[:, i], label='pred')
        #     plt.plot(torque[:, i], label='gt')
        #     plt.legend()
        # plt.show()
    print('rst_mse', rst_mse.mean(axis=(0)).T)
    print(rst_mse.mean(axis=(0, 1)))
    print('rst_rel_err', rst_rel_err.mean(axis=(0)).T)
    print(rst_rel_err.mean(axis=(0, 1)))
    breakpoint()
    

