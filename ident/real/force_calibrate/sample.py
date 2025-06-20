import sys

sys.path.append('../../../..')
sys.path.append('../../..')
from libgex.libgex.libgx11 import Hand
import time
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt

from utils.force_measure.measure import ForceSensor
import os

if __name__ == '__main__':
    default_joint = [np.pi/4, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0,
                    ]
    
    exp_name = 'thumb_vertical_long' #  'thumb_horizontal'  thumb_vertical
    
    # if the directory is not exist, create it

    if not os.path.exists(f'data/{exp_name}'):
        os.makedirs(f'data/{exp_name}')
    
    hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
    hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

    hand.home() # home the hand

    time.sleep(1)
    
    hand.setj(np.array(default_joint)*(180/np.pi))
    
    time.sleep(1.5)

    pos_list = []
    torque_list = []
    t_list = []


    t = time.time()
    
    force_sensor = ForceSensor()
    force_sensor.start()
    
    
    cur_val = 0
    values = []

    # for i in range(int(200)):
    for i in range(int(3000)):
        frame_start = time.time()
        traj_i = hand.getj()

        pos_list.append(hand.getj())
        torque_list.append(hand.get_current())

        t_list.append(time.time() - t)
        
        rst = force_sensor.read_buf_data()
        if rst is not None and len(rst) != 0:
            print(rst)
            cur_val = rst[-1]
        values.append(cur_val)
    
        if time.time() - frame_start < 0.1:
            time.sleep(0.1 - (time.time() - frame_start))
        else:
            print('the frame is too slow')

        print(i, t_list[-1], rst)


    print(time.time() - t)


    pos_list = np.array(pos_list)
    torque_list = np.array(torque_list)
    t_list = np.array(t_list)
    values = np.array(values)

    np.save(f'data/{exp_name}/pos_list.npy', pos_list)
    np.save(f'data/{exp_name}/torque_list.npy', torque_list)
    np.save(f'data/{exp_name}/t_list.npy', t_list)
    np.save(f'data/{exp_name}/force_list.npy', values)

    breakpoint()
