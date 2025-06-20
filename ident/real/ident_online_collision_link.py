
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../..')

from utils.identification.core import IdentFingaer, process_data
from utils.identification.metric import mse, relative_error

# from ident.sim.env import GX11PMEnv
from rviz.rviz_pub import rviz_pub

import scipy.signal as signal
import time

from libgex.libgex.libgx11 import Hand

import rospy


if __name__ == '__main__':
    
    # fix the seed of numpy and random
    np.random.seed(42)
    
    result_map = [
		[0, 1, 2],
		[3, 4, 5, 6],
		[7, 8, 9, 10]
	]

    finger_list = []
    # sim = GX11PMEnv(False)

    
    rviz = rviz_pub()

    time.sleep(1)
    
    for i in range(20):
        print(i)
        rviz.pub_hand_base_link()
        rviz.pub_joint_state()
        rviz.vis_collision_links([])
        # time.sleep(0.1)
        rospy.sleep(0.02)
    # exit()
    
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

        algo = 'ols'

        # get beta
        q, qd, qdd, torque = process_data(q_raw.copy(), torque_raw.copy(), t_list_raw.copy())
        finger.ident_params(q, qd, qdd, torque, algo=algo)

        finger_list.append(finger)



    hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
    hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

    hand.home() # home the hand

    time.sleep(1)

    last_acc = np.zeros(11)
    last_vel = np.zeros(11)
    last_pos = np.zeros(11)
    
    
    is_detection_link = False

    # for i in range(14):
    #     sim.changeVisualShape(i, rgbaColor=[0.3, 0.3, 0.3, 1])   
    # for i in range(10):
    #     sim.step()
    
    # finger_threshold_scale = [
    #     np.array([5, 3, 8]),
    #     np.array([9, 10, 5, 5]),
    #     np.array([5, 8, 5, 4]),
    # ]
    
    finger_fix_threshold_scale = [np.array([0.03 , 0.06, 0.024]), 
                                  np.array([0.1, 0.05, 0.04, 0.035]), 
                                  np.array([0.1, 0.05, 0.05, 0.024])]
    
    last_time = time.time()
    
    print('begin identification')
    joint_tau_data = [[], [], []]
    for t in range(int(1e4)):
        
        # sim.step()
        time_begin = time.time()
        print('-'*50, t)
        # time.sleep(0.1)
        rospy.sleep(0.1)

        # updatae the data
        pos = np.array(hand.getj())
        cur = np.array(hand.get_current())
        cur[cur>32767] = cur[cur>32767]- 65536
        
        cur_pos = pos.copy()
        cur_vel = (cur_pos - last_pos) / (time.time() - last_time)
        cur_acc = (cur_vel - last_vel) / (time.time() - last_time)
        
        for finger_idx, finger in enumerate(finger_list):
            
            torque = cur[finger.joint_mask]/1000

            # pred torque
            pred_torque_i = finger.pred_torque(cur_pos[None], cur_vel[None], cur_acc[None])
            pred_torque_i = np.array(pred_torque_i)
            error = torque[None] - pred_torque_i

            # print(finger_idx, error.shape)
            collision_links = []
            if is_detection_link and t>20:

                collision_joint_idx = np.abs(error)>finger_fix_threshold_scale[finger_idx][None]
                if (collision_joint_idx).sum()>0:

                    joint_idx_forced = np.where(collision_joint_idx)[1]
                    collision_contact_link = np.max(joint_idx_forced)
                    
                    rviz.pub_hand_base_link()
                    rviz.pub_joint_state()
                    
                    collision_links.append(result_map[finger_idx][collision_contact_link])
                
            elif t<=20:
                joint_tau_data[finger_idx].append(error)
                if t==20:
                    finger_joint_tau_mean = []
                    finger_joint_tau_std = []
                    for finger_idx in range(3):
                        # finger_joint_tau_mean[finger_idx] = np.array(joint_tau_data[finger_idx]).mean(axis=(0, 1))
                        # finger_joint_tau_std[finger_idx] = np.array(joint_tau_data[finger_idx]).std(axis=(0, 1))
                        finger_joint_tau_std.append(np.array(joint_tau_data[finger_idx]).std(axis=(0, 1)))
                        finger_joint_tau_mean.append(np.array(joint_tau_data[finger_idx]).mean(axis=(0, 1)))
                    print('Detection the collision link begin!')
                    is_detection_link = True
                    # breakpoint()

            rviz.pub_hand_base_link()
            rviz.pub_joint_state()
            print('collision_links: ', collision_links)
            rviz.vis_collision_links(collision_links)
            
        # update the last data
        cur_torque = cur

        last_pos = pos.copy()
        last_vel = cur_vel
        last_acc = cur_acc
        last_time = time.time()


        print(time.time() - time_begin)
        
