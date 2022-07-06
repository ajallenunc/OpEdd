import os 
import argparse

import numpy as np
import nibabel as nib

from dipy.reconst.shm import real_sym_sh_basis, sph_harm_ind_list
from dipy.io.gradients import read_bvals_bvecs
from dipy.data import get_sphere

import sys 
sys.path.append("../opedd")
from utility import cart2sphere, get_odf_transformation

def fit_voxel_signal(X_i, S_i, sh_order, gamma=0.005):
	"""
	Fit diffusion signal using Descoteaux, 2007.
	Paramters:
		X_i: N_i x 3 numpy array of observation locations on S^2 for i'th subject 
		S_i: N_i x 1 numpy array of noisey signal observations for i'th subject 
		sh_order: int, order of real, symmetric spherical harmonic basis: k = int((sh_order+1)*(sh_order+2)/2)
		gamma: float, regularization strength >= 0
	Output:
		c_hat: k-dimensional array of estimated SH coefficients 
		P: k x k (pseudo) projection matrix
	"""
	X_spherical = cart2sphere(X_i)
	theta = X_spherical[:,0]; phi = X_spherical[:,1]
	B, m, n = real_sym_sh_basis(sh_order, phi, theta)
	R = np.diag(np.power(n, 2)*np.power(n+1, 2))
	c_hat = np.dot(np.linalg.inv(np.dot(B.T, B) + gamma*R) , np.dot(B.T, S_i)) ## for testing purposes only, need a more numerically stable procedure
	P = np.dot(np.dot(B, np.linalg.inv(np.dot(B.T, B) + gamma*R)), B.T)
	return c_hat, P

def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
								description="Estimate historical diffusion signal.")

	parser.add_argument('--data_dir', action='store', required=True,
				   type=str, help='Directory where output to PSC is stored')   

	parser.add_argument('--prior_dir', action='store', required=True,
				   type=str, help='Directory for storing prior.') 

	parser.add_argument('--subject_id', action='store', required=True,
				   type=str, help='Subject id')

	parser.add_argument('--Bval', action='store', required=True,
				   type=float, help='B-value to estimate signal')

	parser.add_argument('--mask', action='store', default=None,
				   type=str, help='Mask (optional)')

	parser.add_argument('--gamma', action='store', default=1e-3,
				   type=float, help='Roughness penalty parameter')
	
	parser.add_argument('--sh_order', action='store', default=8,
				   type=int, help='Order of spherical harmonic basis')

	parser.add_argument('-f', action='store_true', dest='overwrite',
				   help='If set, overwrite files if they already exist.')

	args = parser.parse_args()

	subject_id = args.subject_id
	DATA_DIR = args.data_dir
	PRIOR_DIR = args.prior_dir
	mask = args.mask
	Bval = args.Bval
	gamma = args.gamma
	sh_order = args.sh_order
	K = int((sh_order+1)*(sh_order+2)/2)

	m, n = sph_harm_ind_list(sh_order)
	T_n = get_odf_transformation(n)
	sphere = get_sphere("repulsion724")

	bvals, bvecs = read_bvals_bvecs(os.path.join(DATA_DIR, subject_id, "bvals"), 
									os.path.join(DATA_DIR, subject_id, "bvecs"))
	ixb = np.argwhere((bvals > Bval - 20) & (bvals < Bval + 20)).ravel()
	ix0 = np.argwhere((bvals < 10.)).ravel()

	img = nib.load(os.path.join(DATA_DIR, subject_id, "dwi.nii.gz"))
	data = img.get_fdata()
		
	if mask:
		maskimg = nib.load(os.path.join(DATA_DIR, subject_id, mask))
		maskdata = maskimg.get_fdata()
		masknonzero = np.nonzero(maskdata)
	else:
		masknonzero = np.nonzero(data.shape[:-1])

	X = bvecs[ixb]; Nsamps = X.shape[0]
	signal_tensor = np.zeros((data.shape[0], data.shape[1], data.shape[2], K))
	for ix, iy, iz in zip(masknonzero[0], masknonzero[1], masknonzero[2]):
		S_i = data[ix, iy, iz, ixb].reshape((Nsamps, 1))
		S0_i = data[ix, iy, iz, ix0].mean()
		if S0_i:
			S_i_norm = S_i / S0_i 
			c_hat_i, P_i = fit_voxel_signal(X, S_i_norm, sh_order, gamma)
			signal_ixiyiz = c_hat_i.ravel()
			signal_tensor[ix, iy, iz, :] = signal_ixiyiz

	if not os.path.exists(os.path.join(PRIOR_DIR, "training", subject_id)):
		os.makedirs(os.path.join(PRIOR_DIR, "training", subject_id))
	nib.save(nib.Nifti1Image(signal_tensor, img.affine), 
		os.path.join(PRIOR_DIR, "training", subject_id, "signal_tensor.nii.gz"))

if __name__ == "__main__":
	main()
	