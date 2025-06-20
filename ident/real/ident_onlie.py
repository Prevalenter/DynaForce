
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../..')


from utils.identification.core import IdentFingaer, process_data

from utils.identification.metric import mse, relative_error

import scipy.signal as signal
import time

from libgex.libgex.libgx11 import Hand

if __name__ == '__main__':
    # fix the seed of numpy and random
    np.random.seed(42)
    
    joint_idx = 0
    exp_idx = 0
    joint_mask = np.zeros(11)
    if joint_idx == 0:
        joint_mask[:3] = 1
    else:
        raise NotImplementedError()
        
    finger = IdentFingaer(f'../../data/model/gx11pm_finger{joint_idx+1}.pkl', joint_mask=joint_mask)
    
    q_raw = np.load(f'../../data/ident/pos_list_{joint_idx}_{exp_idx}.npy')
    torque_raw = np.load(f'../../data/ident/torque_list_{joint_idx}_{exp_idx}.npy')
    t_list_raw = np.load(f'../../data/ident/t_list_{joint_idx}_{exp_idx}.npy')

    # get beta
    q, qd, qdd, torque = process_data(q_raw.copy(), torque_raw.copy(), t_list_raw.copy())
    finger.ident_params(q, qd, qdd, torque, algo='ols')
    
    tau_pred = finger.pred_torque(q, qd, qdd)
    
    print(torque.shape, tau_pred.shape)
    error = torque[:, :3] - tau_pred
    
    # for i in range(3): 
    #     plt.subplot(3, 1, i+1)

    #     plt.plot(error[:, i], label='error')
    #     plt.plot(torque[:, i], label='pred')
    #     plt.plot(tau_pred[:, i], label='tau_pred')
    # plt.legend()
    # plt.show()
    
    hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
    hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

    hand.home() # home the hand

    time.sleep(1)

    last_acc = np.zeros(11)
    last_vel = np.zeros(11)
    last_pos = np.zeros(11)
    last_time = time.time()
    while(True):

        time.sleep(0.2)
        
        pos = np.array(hand.getj())
        cur = np.array(hand.get_current())
        cur[cur>32767] = cur[cur>32767]- 65536
        torque = cur[:3]/1000
        print(pos)
        print(cur)

        cur_pos = pos.copy()
        cur_vel = (cur_pos - last_pos) / (time.time() - last_time)
        cur_acc = (cur_vel - last_vel) / (time.time() - last_time)

        # pred torque
        print(cur_pos.shape, cur_vel.shape, cur_acc.shape)
        pred_torque_i = finger.pred_torque(cur_pos[None], cur_vel[None], cur_acc[None])
        pred_torque_i = np.array(pred_torque_i)
        error = torque - pred_torque_i
        print(error)
        # breakpoint()

        cur_torque = cur

        last_pos = cur_pos.copy()
        last_vel = cur_vel
        last_acc = cur_acc
        
        
        
        
        
    
    

