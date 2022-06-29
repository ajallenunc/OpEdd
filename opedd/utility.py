import numpy as np 
from dipy.reconst.shm import real_sym_sh_basis
from scipy.special import legendre
from dipy.segment.mask import applymask
from sklearn.metrics.pairwise import euclidean_distances

class EigenBasis(object):
	def __init__(self, mu, V, rho, sh_order, num_eigen=None, sigma2=0):
		"""
		mu: 1xk numpy array, coefficients for mean function (w.r.t spherical harmonic basis of sh_order)
		V: k x num_eigen_basis, coefficients for eigen-basis (w.r.t spherical harmonic basis of sh_order)
		rho: (num_eigen_basis,) numpy array, eigenvalues 
		sh_order: int, order of spherical harmonic basis 
		sigma2: float, measurement error variance estimate (positive)
		"""
		self.mu = mu 
		self.V = V 
		self.rho = rho
		self.sh_order = sh_order
		self.sigma2 = sigma2
		self.k = int((sh_order+1)*(sh_order+2)/2)
		assert self.k == self.V.shape[0], "Invalid Eigenbasis"
		if num_eigen:
			self.num_eigen = num_eigen
		else:
			self.num_eigen = self.V.shape[1]
		self.Lambda = sum([self.rho[i]*np.dot(self.V[:,i].reshape(self.k,1), self.V[:,i].reshape(1,self.k)) for i in range(self.num_eigen)])
	def mean(self, X):
		"""
		X: sx3 array of cartesian points of evaluation 
		"""
		X_spherical = cart2sphere(X)
		theta = X_spherical[:,0]; phi = X_spherical[:,1]
		B, m, n = real_sym_sh_basis(self.sh_order, phi, theta)
		return np.dot(B, self.mu.T)
	def K(self, X):
		"""
		X: sx3 array of cartesian points of evaluation 
		"""
		C = self.K_pairwise(X, X)
		return C 
	def K_pairwise(self, X1, X2):
		"""
		X1: tx3 numpy array of cartesian points of interest 
		X2: tx3 numpy array of cartesian points of interest 
		"""
		N1 = X1.shape[0]; N2 = X2.shape[0]
		X_spherical_1 = cart2sphere(X1)
		theta_1 = X_spherical_1[:,0]; phi_1 = X_spherical_1[:,1]
		B_1, m_1, n_1 = real_sym_sh_basis(self.sh_order, phi_1, theta_1)
		X_spherical_2 = cart2sphere(X2)
		theta_2 = X_spherical_2[:,0]; phi_2 = X_spherical_2[:,1]
		B_2, m_2, n_2 = real_sym_sh_basis(self.sh_order, phi_2, theta_2)
		C = np.dot(np.dot(B_1, self.Lambda), B_2.T)
		Dx = euclidean_distances(X1, X2)
		zeroix = np.where(Dx == 0)
		if len(zeroix[0]):
			C[zeroix[0], zeroix[1]] = C[zeroix[0], zeroix[1]] + self.sigma2
		return C
	def compute_eigenfunction(self, X, j):
		"""
		X: sx3 array of cartesian points of evaluation 
		j: int defining which eigenfunction to compute
		"""
		X_spherical = cart2sphere(X)
		theta = X_spherical[:,0]; phi = X_spherical[:,1]
		B, m, n = real_sym_sh_basis(self.sh_order, phi, theta)
		return np.dot(B, self.V[:,j])       
	def predict(self, X, coefs=None):
		"""
		X: sx3 array of cartesian points of evaluation
		coefs: num_eigen x 1 array of estimated coefficietns for eigenfunction (if none, return mean prediction)
		"""
		if coefs == None:
			return self.mean(X)
		else:
			Nobs = X.shape[0]
			return self.mean(X) + sum([coefs[kk]*self.compute_eigenfunction(X, kk) for kk in range(self.num_eigen)]).reshape(Nobs, 1)
	def get_eigenvalues(self):
		return self.rho[0:self.num_eigen]

def FPCA(C_hat, sh_order, SORT=False):
	"""
	Functional principal components 
	Parameters:
		C_hat: N x K numpy array, smoothed SH represented signals 
		sh_order: int, order of SH basis 
	"""
	N = C_hat.shape[0]
	mu_hat = C_hat.mean(axis=0).reshape(1,C_hat.shape[1])
	C_hat_centered = C_hat - np.repeat(mu_hat, N, axis=0)
	K = (1/float(N))*np.dot(C_hat_centered.T, C_hat_centered)
	sigma, V = np.linalg.eig(K)
	if SORT:
		idx = sigma.argsort()[::-1]
		sigma = sigma[idx]
		V = V[:, idx]
	return V, sigma, C_hat_centered, mu_hat


def cart2sphere(x):
	r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
	theta = np.arctan2(x[:,1], x[:,0])
	phi = np.arccos(x[:,2]/r)
	return np.column_stack([theta, phi])

def sphere2cart(x):
	theta = x[:,0]
	phi = x[:,1]
	xx = np.sin(phi)*np.cos(theta)
	yy = np.sin(phi)*np.sin(theta)
	zz = np.cos(phi)
	return np.column_stack([xx, yy, zz]) 

def S2hemisphere(x):
	x_copy = np.copy(x)
	x_polar = cart2sphere(x_copy)
	ix = np.argwhere(x_polar[:,1] > np.pi/2).ravel()
	x_copy[ix, :] = -1*x_copy[ix, :] 
	return x_copy

def estimate_rank(EB, alpha=0.95):
	J = EB.num_eigen
	totalvar = np.sum(EB.rho)
	cumvar = 0.
	for k in range(J):
		cumvar = cumvar + EB.rho[k]
		if (cumvar/totalvar) > alpha:
			return k + 1
	return J

def GCV(X, S, sh_order, lam):
	N = len(S)
	gcvs = np.zeros(N)
	for i in range(N):
		Xi = X[i]; Si = S[i].reshape((len(S[i]), 1))
		Ni = Xi.shape[0]
		Xi_spherical = cart2sphere(Xi)
		thetai = Xi_spherical[:,0]; phii = Xi_spherical[:,1]
		Bi, mi, ni = real_sym_sh_basis(sh_order, phii, thetai)
		R = np.diag(np.power(ni, 2)*np.power(ni+1, 2))
		Hi = Bi @ np.linalg.inv(Bi.T @ Bi + lam*R) @ Bi.T 
		Ii = np.identity(Ni)
		SSE_i = np.sum(np.power((Ii - Hi) @ Si, 2))
		residual_space_trace_i = np.sum(np.diag(Ii - Hi))
		gcvs[i] = (SSE_i/Ni)/np.power(residual_space_trace_i/Ni, 2)
	return np.sum(gcvs)

def rankOneUpdate(A_inv, B, D):
	C = B.T
	alpha = 1./(D - C @ A_inv @ B).ravel()
	R_11 = A_inv + alpha*A_inv @ B @ C @ A_inv 
	R_12 = -alpha * A_inv @ B 
	R_21 = -alpha * C @ A_inv
	R_22 = alpha.reshape((1,1))
	return np.block([[R_11,R_12],
				[R_21, R_22]])

def measurement_error_var2(data_b0, mask):
	data_b0_masked = applymask(data_b0, mask)
	data_b0_masked = np.divide(data_b0_masked, np.mean(data_b0_masked, axis = 3)[:,:,:,np.newaxis])
	sigma2_v = np.nanstd(np.where(data_b0_masked != 0, data_b0_masked, np.nan), axis = 3)**2
	sigma2_hat = np.nanmean(sigma2_v)
	return sigma2_hat

def get_odf_transformation(n):
	T = np.zeros((len(n), len(n)))
	for i in range(T.shape[0]):
		P_n = legendre(n[i])
		T[i, i] = P_n(0)
	return T

def get_signal_transformation(n):
	Tinv = np.zeros((len(n), len(n)))
	for i in range(Tinv.shape[0]):
		P_n = legendre(n[i])
		Tinv[i, i] = 1./P_n(0)
	return Tinv