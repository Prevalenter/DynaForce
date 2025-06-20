import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import scipy

from matplotlib import patches

import sys
sys.path.append('../../..')
from utils.identification.algo.wls import weightedLeastSquares, weightedObsservationTorque
from numpy import matlib as mb

# from utils import process_data, gene_robot
# from utils.support_funcs.regr_data_proc import gen_regr_matrices

def updateErrorFullCovariance(W_hash, Y_tau_hash, Beta_LS, omega_old_sqrt, nDOF=6):
	neq1, np1 = W_hash.shape

	residual = np.zeros((nDOF, neq1//nDOF))

	for i in range(nDOF):
		residual[i] = (Y_tau_hash[i::nDOF]-W_hash[i::nDOF]@Beta_LS).T

	omega_new = omega_old_sqrt @ (residual @ residual.T) @ omega_old_sqrt /(neq1-np1)

	return omega_new

def weightFunction(residual, threshold):
	return ( residual<threshold*np.ones(residual.shape) ) * (residual>-threshold*np.ones(residual.shape))

def iterativeReweightedLeastSquares(W, Y_tau, thresholdOmega=1e-2, thresholdWeight=1, nDof=6):
	weights_converged = False
	omega_converged = False

	weight_vector = np.ones((W.shape[0], 1))
	omega = np.eye(nDof)
	# print(W.shape, omega.shape)
	idx = 0
	while weights_converged==False:
		while omega_converged==False:
			idx += 1
			# print(idx)
			ou, os, ov = scipy.linalg.svd(omega)
			os = np.diag(os)
			omega_inv_sqrt = ou @ np.sqrt(np.linalg.pinv(os)) @ ov
			W_star, Y_tau_star = weightedObsservationTorque(W, Y_tau, omega_inv_sqrt, nDof=nDof)

			# W_hash = mb.repmat(weight_vector, 1, W.shape[1])
			W_hash = np.repeat(weight_vector, W.shape[1], axis=1) * W_star
			Y_tau_hash = weight_vector * Y_tau_star

			beta_ls = np.linalg.pinv(W_hash) @ Y_tau_hash

			# update omega
			omega_old = omega
			omega = updateErrorFullCovariance(W_hash, Y_tau_hash,
			                        beta_ls, np.linalg.pinv(omega_inv_sqrt), nDOF=nDof)

			# is converge
			delta_omega = np.linalg.norm(omega-omega_old)
			if delta_omega<thresholdOmega:
				omega_converged = True
			else:
				omega_converged = False

		# update weightvector
		weight_vector_old = weight_vector
		weight_function = weightFunction(Y_tau_hash - W_hash @ beta_ls, 3)*1
		# print(weight_vector.shape, weight_function.shape)
		weight_vector = np.minimum(weight_vector, weight_function)

		# is converge
		delta_weight = np.linalg.norm(weight_vector-weight_vector_old)
		if delta_weight<thresholdWeight:
			weights_converged = True
		else:
			weights_converged = False

		return beta_ls, weight_vector

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
	q, qd, qdd, tau = process_data.get_data('../data/sixJointSinCos2.txt')

	q[1000:2000] *= 2
	# qd[1000:2000] *= 2
	# qdd[1000:2000] *= 2
	# tau[1000:2000] *= 2

	# plt.plot(q.shape)

	W, omega, Q1, R1, rho1 = gen_regr_matrices(rbt, q, qd, qdd, tau)
	Y_tau = omega

	beta_irls, weight_vector = iterativeReweightedLeastSquares(W, Y_tau,
	                                    thresholdOmega=1e-5, thresholdWeight=0.001)
	beta_wls, W_star1, Y_tau_star1 = weightedLeastSquares(W, Y_tau)
	# print(beta_ls)

	#(weight_vector.shape, weight_vector.sum(), weight_vector.sum()/weight_vector.shape[0])
	# plt.plot(weight_vector)
	# plt.show()

	pred_irls = W @ beta_irls
	pred_wls = W @ beta_wls

	plt.figure(figsize=(12, 8))
	for i in range(6):
		plt.subplot(7, 1, i + 1)
		plt.plot(Y_tau[i::6], 'r-.', label='Ground Truth')
		# plt.plot(pred_irls[i::6], 'b-', label='IRLS')
		plt.plot(pred_wls[i::6], 'g-', label='WLS')
		plt.ylabel(f'tau {i+1}')
		# plt.plot(Y_tau[i::6], 'g-', label='Y_tau')
	plt.legend()

	axes = plt.subplot(7, 1, 7)
	# x = np.arange(weight_vector.shape[0])
	# plt.scatter(x, x*0, c=weight_vector, s=1)
	# plt.colorbar(orientation='horizontal')

	plt.plot(weight_vector.reshape((-1, 6)).mean(1))
	rect = patches.Rectangle((1000, -1), 1000, 3, linewidth=1, edgecolor='none',
	                         facecolor='green', alpha=0.4)
	axes.add_patch(rect)
	plt.ylabel('weight')
	plt.ylim(-1, 2)

	plt.subplots_adjust(hspace=0.12)
	plt.show()
