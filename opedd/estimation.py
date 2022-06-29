import numpy as np 
from dipy.reconst.shm import real_sym_sh_basis

def MAP(X, S, EB):
	"""
	MAP estimator under spatial GP prior from arXiv:2102.12526 
	Parameters:
		X: N x 3 numpy array of observation locations on S^2 
		S: N x 1 numpy array of noisey signal observations 
		EB: EigenBasis object, defines the empirical prior 
	Returns estimated coefficients w.r.t. the eigenfunctions. 
	Note: This is a quick inefficient implementation for proof of concept purposes 
	"""
	Gamma_12 = np.zeros((EB.num_eigen, X.shape[0]))
	for kk in range(EB.num_eigen):
		Gamma_12[kk, :] = EB.rho[kk]*EB.compute_eigenfunction(X, kk)
	Gamma_22 = EB.K(X)
	return np.dot(np.dot(Gamma_12, np.linalg.inv(Gamma_22)), S - EB.mean(X))

def roughness_penalized_ridge_estimator(X, S, sh_order, gamma):
	"""
	Ridge estimator w/ reg strength gamma for diffusion signal coefficients from 
			Descoteaux, et. al 2007, Regularized, fast, and robust analytical Q-ball imaging.
	Parameters:
		X: N x 3 numpy array of observation locations on S^2 
		S: N x 1 numpy array of noisey signal observations 
		sh_order: int, order of real, symmetric spherical harmonic basis: k = int((sh_order+1)*(sh_order+2)/2)
		gamma: float, regularization strength >= 0
	"""
	X_spherical = cart2sphere(X_i)
	theta = X_spherical[:,0]; phi = X_spherical[:,1]
	B, m, n = real_sym_sh_basis(sh_order, phi, theta) # B N_i x K ... m == power, length K array , n == order, length K array 
	R = np.diag(np.power(n, 2)*np.power(n+1, 2))
	c_hat = np.dot(np.linalg.inv(np.dot(B.T, B) + gamma*R) , np.dot(B.T, S_i))
	P = np.dot(np.dot(B, np.linalg.inv(np.dot(B.T, B) + gamma*R)), B.T)
	return c_hat, P

