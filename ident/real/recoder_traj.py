import sys

sys.path.append('../../..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
 
hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect(goal_pwm=20) # goal_pwm changes the speed, max 855

hand.home() # home the hand

# for i in range(11):
#     hand.motors[i].init_config(curr_limit=1000, goal_current=1000, goal_pwm=600)
#     hand.motors[i].set_curr_limit(10)

time.sleep(1)

joint_idx = 2
exp_idx = 4

total_time = 10
interal = 0.01
total_timestep = int(total_time / interal)
traj = []
t = time.time()
for i in range(total_timestep):
    traj_i = hand.getj()
    print(i, traj_i)
    traj.append(traj_i)
    time.sleep(interal)
print(time.time() - t)
traj = np.array(traj)
np.save(f'../../data/ident/traj_{joint_idx}_{exp_idx}.npy', traj)


