import numpy as np
import os 
import argparse 
import pickle 
import csv 
import nibabel as nib
import warnings

from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_ind_list
from dipy.reconst.shm import sf_to_sh
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import odf_sh_to_sharp
from dipy.core.sphere import disperse_charges, HemiSphere

import sys 
sys.path.append("/work/users/a/a/aallen1/OptSCAcq/OpEdd/opedd")
from utility import EigenBasis, cart2sphere, GCV, get_odf_transformation
from select_design import GDS, GDS_region
from estimation import MAP, roughness_penalized_ridge_estimator
from compare_util import generate_ESR_design, roughness_penalized_ridge_estimator, get_odf_transformation

# Initialize vars
SUBJECT_DIR = "/work/users/a/a/aallen1/OptSCAcq/test_subs/3150431"
Bval = 2000
M = 20
nv_samps = 100
sh_order = 8
K = int((sh_order+1)*(sh_order+2)/2)
m, n = sph_harm_ind_list(sh_order)
T_n = get_odf_transformation(n)
sphere = get_sphere("repulsion724")


# Load data 
bvals, bvecs = read_bvals_bvecs(os.path.join(SUBJECT_DIR, "bvals"), 
                                os.path.join(SUBJECT_DIR, "bvecs"))
ix0 = np.argwhere((bvals < 10.)).ravel()
ixb = np.argwhere((bvals > Bval - 20) & (bvals < Bval + 20)).ravel()
X = bvecs[ixb] # Bvecs at correct Bval



def main():
    
    """
    Example usage of optimal design code and estimator. 
    Requires 
    1) Prior to be constructed and warped, e.g., using the pipeline outlined in the priors folder.
    2) dwi.nii.gz needs some number >= 1 of diffusion weighted images at `Bval' + >= 2 b0 images
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                                            description="Example usage of OpEdd")
    parser.add_argument('--subject_dir', action='store', required=True,
                                    type=str, help='Path to subject directory.')
    parser.add_argument('--sh_order', action='store', default=8,
                               type=int, help='Order of spherical harmonic basis')
    parser.add_argument('--Bval', action='store', required=True,
                               type=float, help='B-value to estimate signal')
    parser.add_argument('--M', action='store', default=20,
                               type=int, help='Budget')
    parser.add_argument('--rand_seed', action='store', default=0,
                               type=int, help='Random Seed')
    
    
    # Parse Args 
    args = parser.parse_args()
    SUBJECT_DIR = args.subject_dir
    Bval = args.Bval
    M = args.M 
    sh_order = args.sh_order
    rand_seed = args.rand_seed
    K = int((sh_order+1)*(sh_order+2)/2)
    T_n = get_odf_transformation(n)
    gamma = 1e-3
    sphere = get_sphere("repulsion724")

    #### Load data ###
    
    # Subject Image
    img = nib.load(os.path.join(SUBJECT_DIR, "dwi.nii.gz"))
    data = img.get_fdata()

    # Subject Mask
    maskimg = nib.load(os.path.join(SUBJECT_DIR, "mask.nii.gz"))
    maskdata = maskimg.get_fdata()
    masknonzero = np.nonzero(maskdata)
    
    # Bvals and Bvecs
    bvals, bvecs = read_bvals_bvecs(os.path.join(SUBJECT_DIR, "bvals"), 
                                    os.path.join(SUBJECT_DIR, "bvecs"))
    ix0 = np.argwhere((bvals < 10.)).ravel()
    ixb = np.argwhere((bvals > Bval - 20) & (bvals < Bval + 20)).ravel()
    X = bvecs[ixb] # Bvecs at correct Bval

    # Get ESR Sampling Config
    best_config,idx_config,min_energy = generate_ESR_design(X,M,int(2e6),rand_seed)
    
    X_M = best_config
    
    ### Compute ODF & fODF ###
    os.makedirs(os.path.join(SUBJECT_DIR,"SHLS"),exist_ok=True)
    # Initialize Vars 
    SignalTensor = np.zeros((data.shape[0], data.shape[1], data.shape[2], K))
    ODFTensor = np.zeros((data.shape[0], data.shape[1], data.shape[2], K))
    
    for ix, iy, iz in zip(masknonzero[0], masknonzero[1], masknonzero[2]):
        S_v = data[ix, iy, iz, ixb]/(data[ix, iy, iz, ix0].mean())
        S_v = S_v.reshape(len(S_v), 1)
        S_M = S_v[idx_config].reshape(-1,1)
        c_hat_i, p_i = roughness_penalized_ridge_estimator(X_M,S_M,sh_order,gamma)
        signal_ixiyiz = c_hat_i.ravel()
        SignalTensor[ix,iy,iz,:] = signal_ixiyiz
        ODFTensor[ix,iy,iz,:] = T_n @ SignalTensor[ix,iy,iz,:]
        
    
    nib.save(nib.Nifti1Image(SignalTensor,img.affine),
             os.path.join(SUBJECT_DIR,"SHLS",f"shls_{M}_signal_tensor.nii.gz"))
    
    nib.save(nib.Nifti1Image(ODFTensor,img.affine),
             os.path.join(SUBJECT_DIR,"SHLS",f"shls_{M}_odf_tensor.nii.gz"))


    FODFTensor = odf_sh_to_sharp(ODFTensor, sphere, basis="descoteaux07", ratio=3/15., 
                                 sh_order=sh_order,lambda_=1., tau=0.5, r2_term=False) 


    nib.save(nib.Nifti1Image(FODFTensor,img.affine),
             os.path.join(SUBJECT_DIR,"SHLS",f"shls_{M}_fodf_tensor.nii.gz"))

    


if __name__=="__main__":
    main()
