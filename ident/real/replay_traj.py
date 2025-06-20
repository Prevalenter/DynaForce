import sys, os

sys.path.append('..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
import scipy.signal as signal
 
# print( os.listdir('../../data/ident/') )
# exit()
 
hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

hand.home() # home the hand


joint_idx = 0

time.sleep(1)



traj = np.load(f'../../data/ident/traj_2_0.npy')#[::10]



# print(traj.shape)
t = time.time()
for traj_i in traj:
    hand.setj(traj_i)
    time.sleep(0.01)

print(time.time() - t)
