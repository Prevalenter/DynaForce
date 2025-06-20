
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')

from utils.identification.metric import mse, relative_error

# from ident.sim.env import GX11PMEnv
from rviz.rviz_pub import rviz_pub

import scipy.signal as signal
import time

import rospy

from utils.ForceEstimation import HandForce
from libgex.libgex.libgx11 import Hand


    
if __name__ == '__main__':
    
    data_root_dir = '../../data/force_estimation'
    exp_name = 'thumb_vertical'
    
    # fix the seed of numpy and random
    np.random.seed(42)
    
    hand_force = HandForce()
    hand_force.load_default_params()
    # exit()
    
    default_joint = [np.pi/4, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    ]

    rviz = rviz_pub()

    time.sleep(1)

    last_acc = np.zeros(11)
    last_vel = np.zeros(11)
    last_pos = np.zeros(11)

    
    last_time = time.time()
    
    force_fingers = [
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
    ]
    
    # ema_factor = [0.8, 0.8, 0.8]
    ema_factor = [0.0, 0.0, 0.0]
    print('begin identification')
    
    # recoder_pos_list = []
    # recoder_vel_list = []
    # recoder_acc_list = []
    # recoder_torque_list = []
    # recoder_pred_force_list = []

    recoder_pos_list = np.load(f'{data_root_dir}/{exp_name}/pos_list.npy')
    recoder_vel_list = np.load(f'{data_root_dir}/{exp_name}/vel_list.npy')
    recoder_acc_list = np.load(f'{data_root_dir}/{exp_name}/acc_list.npy')
    recoder_torque_list = np.load(f'{data_root_dir}/{exp_name}/torque_list.npy')
    recoder_pred_force_list = np.load(f'{data_root_dir}/{exp_name}/force_pred_list.npy')
    
    print(recoder_pos_list.shape, recoder_vel_list.shape, recoder_acc_list.shape, 
          recoder_torque_list.shape, recoder_pred_force_list.shape)
    # breakpoint()

    for t in range(int(1e3)):
        
        if t%10==0:
            print(t)
        
        # sim.step()
        time_begin = time.time()

        rospy.sleep(0.157)
        
        cur_pos = recoder_pos_list[t]
        
        
        if t>=10:
            # collision_links, collision_link_names, force_pred = hand_force.estimate_force(cur_pos, cur_vel, cur_acc, cur, default_joint)
            force_pred = recoder_pred_force_list[t-10]

            for finger_idx in range(3):
                force_fingers[finger_idx] = force_fingers[finger_idx]*ema_factor[finger_idx] +\
                                                force_pred[finger_idx]*(1-ema_factor[finger_idx])

                # if with fingertip force, visualize
                if np.linalg.norm(force_fingers[finger_idx])>0.04:

                    if finger_idx == 0:
                        link_name = "hand_/Link12"
                    elif finger_idx == 1:
                        link_name = "hand_/Link13"
                    elif finger_idx == 2:
                        link_name = "hand_/Link14"
                        
                    trans = rviz.get_relative_position('hand_/base_link', link_name)
                    rviz.pub_hand_base_link()
                    base_link_name = 'hand_/base_link'

                    rviz.vis_force(link_name=base_link_name, force=-force_pred[finger_idx], scale=0.5, lifetime=0.4, start_bias=trans[0])
                    # rviz.vis_collision_links(collision_links)   
            pos_pub = cur_pos
            
            rviz.pub_joint_state(pos_pub)
            
            # recoder_pred_force_list.append(force_pred.copy())
            
        # # update the last data
        # cur_torque = cur

        # last_pos = cur_pos.copy()
        # last_vel = cur_vel
        # last_acc = cur_acc
        # last_time = time.time()

        # if t%100==99:
        #     print('save the data')


        #     # breakpoint()