import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.io import loadmat

import sys
sys.path.append('..')

# from utils import process_data, gene_robot
# from utils.support_funcs.regr_data_proc import gen_regr_matrices

def getErrorCovariance(W, tau, nDof=6):
	sig2_error = np.zeros((nDof, 1))
	for i in range(nDof):
		_, ri = scipy.linalg.qr(W[i::nDof], mode='economic')
		diag_ri = np.abs(np.diag(ri))
		r_idx = np.nonzero(diag_ri >= 1)

		W_reduced_i = W[i::nDof, r_idx[0]]
		# print(W_reduced_i.shape)
		neq1, np1 = W_reduced_i.shape

		beta_ls = np.linalg.pinv(W_reduced_i) @ tau[i::nDof]

		residual =tau[i::nDof] - W_reduced_i @ beta_ls
		norm_res = np.linalg.norm(residual)

		sig2_error[i] = norm_res**2/(neq1-np1)
	return sig2_error

def weightedObsservationTorque(W, tau, Q, nDof=6):
	W_star = np.zeros(W.shape)
	Y_tau_star = np.zeros(tau.shape)
	# breakpoint()
	for i in range(W.shape[0]//nDof):
		W_star[nDof*i:nDof*(i+1)] = Q @ W[nDof*i:nDof*(i+1)]
		Y_tau_star[nDof*i:nDof*(i+1)] = Q @ tau[nDof*i:nDof*(i+1)]
	return W_star, Y_tau_star

def weightedLeastSquares(W, tau, nDof=6):
	sig2_error1 = getErrorCovariance(W, tau, nDof=nDof)
	omega_inv_sqrt = np.diag(np.sqrt(1./sig2_error1)[:, 0])

	W_star, Y_tau_star = weightedObsservationTorque(W, tau, omega_inv_sqrt, nDof)

	beta_ls = np.linalg.pinv(W_star) @ Y_tau_star

	return beta_ls, W_star, Y_tau_star

if __name__ == '__main__':
	# W = loadmat('../data/wls_matlab/W.mat')['W']
	# W_star = loadmat('../data/wls_matlab/W_star.mat')['W_star']
	# Y_tau = loadmat('../data/wls_matlab/Y_tau.mat')['Y_tau']
	# Y_tau_star = loadmat('../data/wls_matlab/Y_tau_star.mat')['Y_tau_star']
	# sig2_error_gt = loadmat('../data/wls_matlab/sig2_error.mat')['sig2_error']
	# print(W.shape, W_star.shape, Y_tau.shape, Y_tau_star.shape, sig2_error_gt.shape)

	# data = np.load('../data/sixJointSinCos2.npy', allow_pickle=True).item()
	# W = data['W'].reshape(-1, 50)
	# Y_tau = data['tau'].reshape(-1, 1)
	# print(W.shape, Y_tau.shape)
	rbt = gene_robot.get_robot('../data/estun/estun_model.pkl')
	# q, qd, qdd, tau = process_data.get_sim_data()
	q, qd, qdd, tau = process_data.get_data('../data/sixJointSinCos2.txt')
	print(q.shape, qd.shape, qdd.shape, tau.shape)
	W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)


	beta_wls, W_star1, Y_tau_star1 = weightedLeastSquares(W, omega)
	pred_wls1 = W @ beta_wls

	beta_ols = np.linalg.pinv(W) @ omega
	pred_ols1 = W @ beta_ols

	print(pred_wls1.shape, omega.shape)

	plt.figure(figsize=(12, 8))
	for i in range(6):
		plt.subplot(6, 1, i + 1)
		plt.plot(omega[i::6], 'r-.', label='Ground Truth')
		plt.plot(pred_wls1[i::6], 'b-', label='WLS')
		plt.plot(pred_ols1[i::6], 'g-', label='OLS')
		# plt.plot(Y_tau[i::6], 'g-', label='Y_tau')
		plt.legend()
	plt.subplots_adjust(hspace=0.1)
	plt.show()


