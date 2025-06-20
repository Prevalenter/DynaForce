import numpy as np

def LeastSquares(W, tau):
	return np.linalg.pinv(W) @ tau