
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
	# rbt = get_robot('../../data/model/estun_model_mdh.pkl')

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

	# breakpoint()

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
 



	'''
	import numpy as np
	from matplotlib import pyplot as plt

	import sys
	sys.path.append('..')

	from utils import process_data, gene_robot
	from utils.support_funcs.regr_data_proc import gen_regr_matrices

	def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau

	if __name__ == '__main__':

	rbt = gene_robot.get_robot('../data/estun/estun_model_mdh.pkl')



	q, qd, qdd, tau = np.load('../data/data_train/data.npy')
	W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)

	# for i in range(6):
	# 	plt.plot(qd[:, i])
	# plt.show()
	# breakpoint()
	# print(omega.reshape((-1, 6)).shape, tau.shape, (omega.reshape((-1, 6))==tau).sum())

	print(W.shape, omega.shape, Q1.shape, R1.shape, rho1.shape)

	beta_ols = LeastSquares(W, omega)
	pred_ols = W @ beta_ols
	print(beta_ols)

	measure = omega.reshape((-1, 6))
	pred = pred_ols.reshape((-1, 6))

	np.save('../data/data_train/pred_ols.npy', pred)

	for i in range(6):
		plt.subplot(6, 1, 1+i)
		plt.plot(pred[:, i], label='pred')
		plt.plot(measure[:, i], label='measure')
		plt.plot(pred[:, i]-measure[:, i], label='error', c='r')
	# plt.legend()
	plt.show()
	'''


