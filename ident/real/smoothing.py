import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

data = np.load('traj_0.npy')[:, 0]

# print(data.shape)

# smoothing the data using a moving average
window_size = 100
data = signal.savgol_filter(data, window_size, 3)


plt.plot(data)
plt.show()  


