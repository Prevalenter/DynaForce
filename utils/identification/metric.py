import numpy as np
from numpy import mean
from numpy.linalg import norm

def mse(tau_gt, tau_pred):
	return np.linalg.norm(tau_pred - tau_gt)**2 / tau_gt.shape[0]

def relative_error(tau_gt, tau_pred):
	return np.linalg.norm(tau_pred - tau_gt) / np.linalg.norm(tau_gt+1e-1) * 100




def analyse(W, omega, R1, beta):
	p = dict()

	n = W.shape[0]

	omega_norm = norm(omega)
	omega_mean = mean(omega)

	p['err'] = norm(omega - W * beta)
	p['merr'] = p['err'] / n

	p['se'] = p['err' ]**2
	p['mse'] = p['se' ] /( n -W.shape[1])
	p['rmse'] = p['mse' ]**0.5

	C = p['mse'] * (R1.T * R1).I
	p['sd'] = np.sqrt(C.diagonal()).T
	p['sd%'] = 100. * p['sd'] / np.abs(beta)

	p['relerr'] = p['err' ]/ omega_norm
	p['relerr%'] = p['relerr' ] *100.

	p['1-r2'] = p['err' ]**2 / norm(omega - omega_mean )**2
	p['r2'] = 1 - p['1-r2']

	return p