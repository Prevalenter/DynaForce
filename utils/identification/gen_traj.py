import numpy as np
import matplotlib.pyplot as plt
import sys sys.path.append('..')
import numpy as np import sympy import math
from utils import process_data, gene_robot
from utils.dvrk.trajectory_optimization.traj_plotter import TrajPlotter
from utils.dvrk.trajectory_optimization.fourier_traj import FourierTraj

if __name__ == '__main__':
	x = np.random.random(dof * (1 + 2 * order))
	q, dq, ddq = fourier_traj1.fourier_base_x2q(x)

