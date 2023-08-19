import numpy as np
import os 
import argparse 
import pickle 
import csv 
import nibabel as nib

from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_ind_list
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.csdeconv import odf_sh_to_sharp

import sys 
sys.path.append("/work/users/a/a/aallen1/OptSCAcq/OpEdd/opedd")
from utility import EigenBasis, cart2sphere, GCV, get_odf_transformation
from select_design import GDS, GDS_region
from estimation import MAP, roughness_penalized_ridge_estimator

np.random.seed(0)

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
    parser.add_argument('--n_voxel_samps', action='store', default=1000,
                               type=int, help='Number of voxels in mask to sample for common design')
    parser.add_argument('--M', action='store', default=20,
                               type=int, help='Budget')


    # global parameters 
    args = parser.parse_args()
    SUBJECT_DIR = args.subject_dir
    Bval = args.Bval
    M = args.M 
    nv_samps = args.n_voxel_samps
    sh_order = args.sh_order
    K = int((sh_order+1)*(sh_order+2)/2)
    m, n = sph_harm_ind_list(sh_order)
    T_n = get_odf_transformation(n)
    sphere = get_sphere("repulsion724")

    # load EB object 
    with open(os.path.join(SUBJECT_DIR,"EB_object.pkl"), "rb") as pklfile:
            EB = pickle.load(pklfile)

    # load data 
    bvals, bvecs = read_bvals_bvecs(os.path.join(SUBJECT_DIR, "bvals"), 
                                                                    os.path.join(SUBJECT_DIR, "bvecs"))
    ix0 = np.argwhere((bvals < 10.)).ravel()
    ixb = np.argwhere((bvals > Bval - 20) & (bvals < Bval + 20)).ravel()

    img = nib.load(os.path.join(SUBJECT_DIR, "dwi.nii.gz"))
    data = img.get_fdata()

    X = bvecs[ixb]

    # load mask 
    maskimg = nib.load(os.path.join(SUBJECT_DIR, "mask.nii.gz"))
    maskdata = maskimg.get_fdata()
    masknonzero = np.nonzero(maskdata)

    ## Create a single design for all voxels in MASK 
    EBkeys = list(EB.keys())
    EBlst = [EB[EBkeys[ix]] for ix in np.random.choice(len(EBkeys), size=nv_samps, replace=False, p=None)]
    Xix_common = GDS_region(X, EBlst, M)
    X_M = X[Xix_common,:]

    print("Finished Setup")
    
    ## Select a voxel specific design and then estimate signal via MAP under historical prior 
    ## Compute ODF via FRT and fODF via deconvolution 

    os.makedirs(os.path.join(SUBJECT_DIR,"OpEdd"),exist_ok=True)

    print("Entering ODF Tensor Loop")
    SignalTensor = np.zeros((data.shape[0], data.shape[1], data.shape[2], K))
    ODFTensor = np.zeros((data.shape[0], data.shape[1], data.shape[2], K))
    for ix, iy, iz in zip(masknonzero[0], masknonzero[1], masknonzero[2]):
        if (ix, iy, iz) in EB:
            S_v = data[ix, iy, iz, ixb]/(data[ix, iy, iz, ix0].mean())
            S_v = S_v.reshape(len(S_v), 1)
            EB_v = EB[(ix, iy, iz)]
            V_hat_v = EB_v.V[:, 0:EB_v.num_eigen]
            ## uncomment for voxel specific designs 
            #Xix_gds_v, mise_v = GDS(X, EB_v, M)
            #X_M = X[Xix_gds_v,:]
            #S_M = S_v[Xix_gds_v].reshape(-1,1)
            S_M = S_v[Xix_common].reshape(-1,1)
            c_hat_eigen = MAP(X_M, S_M, EB_v)
            c_hat_sh = EB_v.mu.T + V_hat_v @ c_hat_eigen
            SignalTensor[ix, iy, iz, :] = c_hat_sh.ravel()
            ODFTensor[ix, iy, iz, :] = T_n @ SignalTensor[ix, iy, iz, :]

    nib.save(nib.Nifti1Image(ODFTensor,img.affine),os.path.join(SUBJECT_DIR,"OpEdd",f"{Bval}_{M}_odf_est.nii.gz"))

    print("Finished ODF Tensor Loop. Startin FODF Calculation")

    FODFTensor = odf_sh_to_sharp(ODFTensor,sphere,basis="descoteaux07",ratio=3/15., sh_order=sh_order,lambda_=1., tau=0.5, r2_term=False)

    print("Finished FODF Tensor Calculation")
    
    nib.save(nib.Nifti1Image(FODFTensor,img.affine),os.path.join(SUBJECT_DIR,"OpEdd",f"{Bval}_{M}_fodf_est.nii.gz"))

if __name__=="__main__":
    main()
