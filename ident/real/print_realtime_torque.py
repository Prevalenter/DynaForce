import sys

sys.path.append('../../..')

from libgex.libgex.libgx11 import Hand
import time
import numpy as np
import scipy.signal as signal
 
hand = Hand(port='/dev/ttyACM0') # COM* for Windows, ttyACM* or ttyUSB* for Linux
hand.connect(goal_pwm=600) # goal_pwm changes the speed, max 855

hand.home() # home the hand

time.sleep(1)

while(True):
    cur = np.array(hand.get_current())
    cur[cur>32767] = cur[cur>32767]- 65536

    print(cur)
    
    time.sleep(0.2)
    
