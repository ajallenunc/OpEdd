import numpy as np 
import copy 
import operator 

from utility import rankOneUpdate

def GDS(X, EB, M):
	"""
	Greedy-designer Algorithm 1: Selection based on maximizing the contional MISE
	Input: 
		X: Nx3 cartesian point set undergoing selection 
		EB: eigenbasis object 
		M: max budget for number of points to select 
	Returns: 
		Xix: length M array of row indices of X defining the best M sample locations on the grid. Note, for any K < M, the first K indices of this list provide the best K locations.  
		ordered row indices of sequential design selection
		mise: length M array whose m'th element is the estimated MISE for the size m-design
	"""
	Xix = np.zeros(M, dtype=int)
	maxVals = np.zeros(M)
	N = X.shape[0]
	Lambda = np.diag(EB.get_eigenvalues()) 
	indices = np.arange(N)
	KK = EB.num_eigen
	Psi_X = np.array([EB.compute_eigenfunction(X, k) for k in range(KK)]).T
	Gamma_X = EB.K(X)
	N = X.shape[0]
	vals = np.zeros(N)
	mise = np.zeros(M)
	trLambda = np.sum(np.diag(Lambda))
	for r in range(M):
		if r == 0:
			obj_func = np.diag(Psi_X @ Lambda @ Lambda @ Psi_X.T)
			max_ix = np.argmax(obj_func)
			Psi_current = Psi_X[max_ix, :].reshape((1, KK))
			Gamma_inv_current = 1./Gamma_X[max_ix, max_ix].reshape(1,1)
		else:
			avail_indices = np.setdiff1d(indices, Xix[0:r])
			obj_func = {}
			for i in avail_indices:
				Psi_i = np.vstack((Psi_current, Psi_X[i, :].reshape((1, KK))))
				gamma_i = Gamma_X[Xix[0:r], i].reshape((r, 1))
				Gamma_i_inv = rankOneUpdate(Gamma_inv_current, gamma_i, Gamma_X[i,i].reshape((1,1)))
				obj_func[i] = np.trace(Lambda @ Psi_i.T @ Gamma_i_inv @ Psi_i @ Lambda)
			max_ix = max(obj_func.items(), key=operator.itemgetter(1))[0]
			Psi_current = np.vstack((Psi_current, Psi_X[max_ix, :].reshape((1, KK))))
			Gamma_inv_current = rankOneUpdate(Gamma_inv_current, 
												Gamma_X[Xix[0:r], max_ix].reshape((r, 1)), 
												Gamma_X[max_ix,max_ix].reshape((1,1)))
		mise[r] = trLambda - obj_func[max_ix]
		Xix[r] = max_ix 
	return Xix, mise

def GDS_region(X, EBlst, M):
	"""
	Algorithm 1 augmented for region of interest
	Input: 
		X: Nx3 cartesian point set undergoing selection 
		EBlst: List of eigenbasis object corresponding to region of interes
		M: max budget for number of points to select 
	Returns: 
		Xix: length M array of row indices of X defining the best M sample locations on the grid. Note, for any K < M, the first K indices of this list provide the best K locations.  
		ordered row indices of sequential design selection
	"""
	Xix = np.zeros(M, dtype=int)
	maxVals = np.zeros(M)
	N = X.shape[0]
	indices = np.arange(N)
	Nvoxels = len(EBlst)
	KKmap = {}; Lambda_map = {}; Psi_map = {}; Gamma_map = {}
	for v, EB in enumerate(EBlst):
		KKmap[v] = EB.num_eigen
		Lambda_map[v] = EB.get_eigenvalues() ## length K array 
		Psi_map[v] = np.array([EB.compute_eigenfunction(X, k) for k in range(KKmap[v])]).T
		Gamma_map[v] = EB.K(X)
	for r in range(M): ## Note: Need to add in weights
		obj_func = np.zeros((N, Nvoxels))
		if r == 0:
			for v in range(Nvoxels):
				Psi_X_v = Psi_map[v]
				Lambda_v = np.diag(Lambda_map[v])
				Gamma_X_v = Gamma_map[v]
				obj_func[:, v] = np.diag(Psi_X_v @ Lambda_v @ Lambda_v @ Psi_X_v.T)
			max_ix = np.argmax(np.sum(obj_func, 1))
			Psi_current = {}; Gamma_inv_current = {}
			for v in range(Nvoxels):
				Psi_X_v = Psi_map[v]
				Gamma_X_v = Gamma_map[v]
				KK_v = KKmap[v]
				Psi_current[v] = Psi_X_v[max_ix, :].reshape((1, KK_v))
				Gamma_inv_current[v] = 1./Gamma_X_v[max_ix, max_ix].reshape(1,1)
		else:
			avail_indices = np.setdiff1d(indices, Xix[0:r])
			for i in avail_indices:
				for v in range(Nvoxels):
					Psi_X_v = Psi_map[v]
					Lambda_v = np.diag(Lambda_map[v])
					Gamma_X_v = Gamma_map[v]
					Psi_current_v = Psi_current[v]
					Gamma_inv_current_v = Gamma_inv_current[v]
					KK_v = KKmap[v]
					Psi_v_i = np.vstack((Psi_current_v, Psi_X_v[i, :].reshape((1, KK_v))))
					gamma_v_i = Gamma_X_v[Xix[0:r], i].reshape((r, 1))
					Gamma_v_i_inv = rankOneUpdate(Gamma_inv_current_v, gamma_v_i, Gamma_X_v[i,i].reshape((1,1)))
					obj_func[i, v] = np.trace(Lambda_v @ Psi_v_i.T @ Gamma_v_i_inv @ Psi_v_i @ Lambda_v)
			max_ix = np.argmax(np.sum(obj_func, 1))
			for v in range(Nvoxels):
				Psi_X_v = Psi_map[v]
				Psi_current_v = Psi_current[v]
				Gamma_X_v = Gamma_map[v]
				Gamma_inv_current_v = Gamma_inv_current[v]
				KK_v = KKmap[v]
				Psi_current[v] = np.vstack((Psi_current_v, Psi_X_v[max_ix, :].reshape((1, KK_v))))
				Gamma_inv_current[v] = rankOneUpdate(Gamma_inv_current_v, 
												Gamma_X_v[Xix[0:r], max_ix].reshape((r, 1)), 
												Gamma_X_v[max_ix,max_ix].reshape((1,1)))
		Xix[r] = max_ix
	return Xix

