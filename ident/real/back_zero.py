import sys

sys.path.append('..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
 
hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

hand.home() # home the hand

# for i in range(11):
#     hand.motors[i].init_config(curr_limit=1000, goal_current=1000, goal_pwm=600)
#     hand.motors[i].set_curr_limit(10)

time.sleep(1)


