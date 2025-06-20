
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

    hand = Hand(port='/dev/ttyACM0', kin=False) # COM* for Windows, ttyACM* or ttyUSB* for Linux
    hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855
    
    hand.home() # home the hand
    time.sleep(1)
    hand.setj(np.array(default_joint)*(180/np.pi)) # set the joint position
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

    for t in range(int(1e4)):
        # sim.step()
        time_begin = time.time()

        rospy.sleep(0.1)

        # updatae the data
        pos = np.array(hand.getj())

        cur = np.array(hand.get_current())
        cur[cur>32767] = cur[cur>32767]- 65536
        torque_norm =1000
        cur = cur*hand_force.urdf_sign[0]/torque_norm

        cur_pos = pos.copy()*(np.pi/180)*hand_force.urdf_sign[0]
        cur_vel = (cur_pos - last_pos) / (time.time() - last_time)
        cur_acc = (cur_vel - last_vel) / (time.time() - last_time)
        
        if t>=10:
            collision_links, collision_link_names, force_pred = hand_force.estimate_force(cur_pos, cur_vel, cur_acc, cur, default_joint)

            for finger_idx in range(3):
                force_fingers[finger_idx] = force_fingers[finger_idx]*ema_factor[finger_idx] +\
                                                force_pred[finger_idx]*(1-ema_factor[finger_idx])

                # if with fingertip force, visualize
                if np.linalg.norm(force_fingers[finger_idx])>0.04:
                    link_name = collision_link_names[finger_idx]
                    trans = rviz.get_relative_position('hand_/base_link', link_name)
                    rviz.pub_hand_base_link()
                    base_link_name = 'hand_/base_link'
                    # print(trans[0])
                    # breakpoint()
                    rviz.vis_force(link_name=base_link_name, force=force_pred[finger_idx], scale=0.5, lifetime=0.4, start_bias=trans[0])
                    rviz.vis_collision_links(collision_links)   
            pos_pub = cur_pos
            
            rviz.pub_joint_state(pos_pub)

            
        # update the last data
        cur_torque = cur

        last_pos = cur_pos.copy()
        last_vel = cur_vel
        last_acc = cur_acc
        last_time = time.time()


