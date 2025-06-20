import sys

sys.path.append('../../..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
import scipy.signal as signal
 
hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

hand.home() # home the hand


joint_idx = 2
exp_idx = 4


time.sleep(1)
traj = np.load(f'../../data/ident/traj_{joint_idx}_{exp_idx}.npy')[::2]

window_size = 100
traj[:, 0] = signal.savgol_filter(traj[:, 0], window_size, 3)


pos_list = []
torque_list = []
t_list = []
# print(traj.shape)
t = time.time()
for traj_i in traj:
    hand.setj(traj_i)
    pos_list.append(hand.getj())
    torque_list.append(hand.get_current())
    # time.sleep(0.01)
    t_list.append(time.time() - t)


print(time.time() - t)


pos_list = np.array(pos_list)
torque_list = np.array(torque_list)
t_list = np.array(t_list)

np.save(f'../../data/ident/pos_list_{joint_idx}_{exp_idx}.npy', pos_list)
np.save(f'../../data/ident/torque_list_{joint_idx}_{exp_idx}.npy', torque_list)
np.save(f'../../data/ident/t_list_{joint_idx}_{exp_idx}.npy', t_list)