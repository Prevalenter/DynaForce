import numpy as np

import sys
sys.path.append('..')
from utils.identification.algo.ols import LeastSquares

def inertial_ransac(W, omega, tau, nDof, n_base, n_data=100, n_p=30, nI=100, max_distance=0.3):

	n_i = omega.shape[0] // (n_data*nDof)

	best_num_inlier = 0
	beta_best = None

	index = np.arange(n_data)

	W = np.array(W)
	omega = np.array(omega)

	W1_reshaped = W[:n_i * nDof * n_data].reshape((-1, n_i * nDof, n_base))
	omega1_reshaped = omega[:n_i * nDof * n_data].reshape((-1, n_i * nDof, 1))

	for epoch in range(nI):
		index_sampled = np.random.choice(index, n_p, replace=False)
		index_sampled.sort()

		W_sampled = W1_reshaped[index_sampled].reshape((-1, n_base))
		omega_sampled = omega1_reshaped[index_sampled].reshape((-1, 1))

		beta_sampled = LeastSquares(W_sampled, omega_sampled)

		pred_sampled = (W @ beta_sampled).reshape((-1, nDof))

		res = np.linalg.norm(pred_sampled - tau, axis=1)
		num_inlier = (res < max_distance).sum()

		if num_inlier > best_num_inlier:
			best_num_inlier = num_inlier
			beta_best = beta_sampled.copy()

			# print('get best', best_num_inlier)

	return beta_best