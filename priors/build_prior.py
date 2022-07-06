import os 
import argparse

import numpy as np 
import nibabel as nib 
from dipy.segment.mask import applymask

import sys 

def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
								description="Build prior from warped diffusion signal.")

	parser.add_argument('--prior_dir', action='store', required=True,
				   type=str, help='Directory where training data is to be written') 

	parser.add_argument('--sh_order', action='store', default=8,
				   type=int, help='Order of spherical harmonic basis')

	args = parser.parse_args()
	PRIOR_DIR = args.prior_dir
	sh_order = args.sh_order
	K = int((sh_order+1)*(sh_order+2)/2)

	template_img = nib.load(os.path.join(PRIOR_DIR, "template", "template.nii.gz"))
	template = template_img.get_fdata()
	nx, ny, nz = template.shape
	N_train = len(os.listdir(os.path.join(PRIOR_DIR,"training")))
	coef_tensors = np.zeros((nx, ny, nz, K, N_train)) ## ~ 153GB
	mask = np.ones((nx, ny, nz))

	for i, subject_id in enumerate(os.listdir(os.path.join(PRIOR_DIR,"training"))):
		coef_tensor = nib.load(os.path.join(PRIOR_DIR, "training", subject_id, "signal_tensor_registered.nii.gz")).get_fdata()
		coef_tensor_mask = np.sum(np.abs(coef_tensor), axis=3)
		coef_tensor_mask[coef_tensor_mask != 0] = 1
		mask = mask * coef_tensor_mask
		coef_tensors[:,:,:,:,i] = coef_tensor

	mu_hat = np.mean(coef_tensors, axis=4)
	mu_hat_masked = applymask(mu_hat, mask)
	nib.save(nib.Nifti1Image(mu_hat_masked, template_img.affine), 
		os.path.join(PRIOR_DIR, "template", "mean_signal.nii.gz"))

	Cov_tensor = np.zeros((nx, ny, nz, int(K*(K+1)/2))) ## ~ 16GB 
	masknonzero = np.nonzero(mask)
	for ix, iy, iz in zip(masknonzero[0], masknonzero[1], masknonzero[2]):
		coefs_v = coef_tensors[ix, iy, iz, :, :].T
		mu_hat_v = mu_hat_masked[ix, iy, iz, :].reshape(1,K)
		coefs_v_centered = coefs_v - np.repeat(mu_hat_v, N_train, axis=0)
		Cov_hat = (1./N_train)*coefs_v_centered.T @ coefs_v_centered
		s, V = np.linalg.eig(Cov_hat)
		Cov_hat_log = V @ np.diag(np.log(s)) @ V.T 
		Cov_tensor[ix, iy, iz, :] = Cov_hat_log[np.triu_indices(K)]
	
	nib.save(nib.Nifti1Image(Cov_tensor, template_img.affine),
		os.path.join(PRIOR_DIR, "template", "log_cov_signal.nii.gz"))

if __name__ == "__main__":
	main()
