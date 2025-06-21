
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append('../..')

from utils.identification.gene_robot import get_robot
from utils.identification.support_funcs.regr_data_proc import gen_regr_matrices

def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau

if __name__ == '__main__':

	rbt = get_robot('../../data/model/gx11pm_finger1.pkl')

	# load data
	data = np.load('sensor.npy').reshape((-1, 11, 3))
	print(data.shape)

	q = data[:, :, 0]
	qd = data[:, :, 1]
	tau = data[:, :, 2]
 

	qdd = (qd[1:]-qd[:-1])*240
	q = q[200+1:, :3]
	qd = qd[200+1:, :3]
	tau = tau[200+1:, :3]
	qdd = qdd[200:, :3]

	print(q.shape, qd.shape, qdd.shape, tau.shape)


	W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
	print(W.shape, omega.shape)

	beta_ols = LeastSquares(W, omega)
	pred_ols = (W @ beta_ols).reshape((-1, 3))
	print(beta_ols)

	# plot in 3 subplots
	for i in range(3):
		plt.subplot(3, 1, i+1)
		plt.plot(pred_ols[:, i], label='pred')
		plt.plot(tau[:, i], label='gt')
		

	plt.legend()
	plt.show()
