import numpy as np
import os 
import argparse 
import pickle 
import nibabel as nib 

from dipy.data import get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import applymask

import sys 
sys.path.append("../opedd")
from utility import EigenBasis, measurement_error_var2

np.random.seed(0)

def main():
	"""
	Builds the eigenbasis objects used for signal estimation and/or design selection.
	Requires:
		1) prior warped onto the subjects brain
		2) >= 2 b0 images for noise variance estimation 
	"""
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
								description="Create EigenBasis objects")

	parser.add_argument('--subject_dir', action='store', required=True,
					type=str, help='Path to subject directory.')
	parser.add_argument('--sh_order', action='store', default=8,
				   type=int, help='Order of spherical harmonic basis')

	# global parameters 
	args = parser.parse_args()
	SUBJECT_DIR = args.subject_dir
	sh_order = args.sh_order
	K = int((sh_order+1)*(sh_order+2)/2)

	# load prior 
	logCovimg = nib.load(os.path.join(SUBJECT_DIR, "log_cov_func_warped.nii.gz"))## (nx, nx, nz, K*(K+1)/2)
	Muimg = nib.load(os.path.join(SUBJECT_DIR, "mean_func_warped.nii.gz"))## (nx, nx, nz, K)
	logCov = logCovimg.get_fdata()
	Mu = Muimg.get_fdata()

	# load b0 data 
	bvals, bvecs = read_bvals_bvecs(os.path.join(SUBJECT_DIR, "bvals"), 
									os.path.join(SUBJECT_DIR, "bvecs"))
	ix0 = np.argwhere((bvals < 10.)).ravel()

	img = nib.load(os.path.join(SUBJECT_DIR, "dwi.nii.gz"))
	data = img.get_fdata()

	# load mask 
	maskimg = nib.load(os.path.join(SUBJECT_DIR, "mask.nii.gz"))
	maskdata = maskimg.get_fdata()

	# estimate measurement error variance 
	sigma2_hat = measurement_error_var2(data[:, :, :, ix0], maskdata)

	masknonzero = np.nonzero(maskdata)

	# construct eigenbasis object for all voxels in mask 
	EBdata = {}
	for ix, iy, iz in zip(masknonzero[0], masknonzero[1], masknonzero[2]):
		logCov_ixiyiz = logCov[ix, iy, iz, :]
		mu_ixiyiz = Mu[ix, iy, iz, :].reshape(1, K)
		if np.sum(logCov_ixiyiz) and np.sum(mu_ixiyiz): ## check if data is present
			logC_v = np.zeros((K, K))
			logC_v[np.triu_indices(K)] = logCov_ixiyiz
			logC_v = np.where(logC_v,logC_v,logC_v.T)
			logrho, V = np.linalg.eig(logC_v)
			idx = logrho.argsort()[::-1]
			logrho = logrho[idx]
			V = V[:, idx]
			rho = np.exp(logrho)
			EB = EigenBasis(mu_ixiyiz, V, rho, sh_order, num_eigen=K-1, sigma2=sigma2_hat)
			EBdata[(ix, iy, iz)] = EB

	with open(os.path.join(SUBJECT_DIR, "EB_object.pkl"), "wb") as pklfile:
		pickle.dump(EBdata, pklfile)

if __name__ == "__main__":
	main()