#!/bin/bash
#SBATCH -t 01:00:00 
#SBATCH -c 2 
#SBATCH -n 1 
#SBATCH --mem-per-cpu=128GB
#SBATCH --array=0-30

module load ants
ANTSPATH=/software/ants/2.1.0
export ANTSPATH

PRIOR_DIR=/scratch/wconsagr/bh_data2/priors/HCP_2000

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
cd /scratch/wconsagr/bh_data2/SMS_New
arr=(*)
SUBJECTID=${arr[$SLURM_ARRAY_TASK_ID]}

DATA_DIR=/scratch/wconsagr/bh_data2/SMS_New/${SUBJECTID}

antsApplyTransforms -d 3 -e 3 -i ${PRIOR_DIR}/mean.nii.gz \
-r ${DATA_DIR}/dwi_psc_connectome/diffusion/dti/fa.nii.gz -t [ ${DATA_DIR}/prior_estimation/fa_warped0GenericAffine.mat , 1 ] \
-t ${DATA_DIR}/prior_estimation/fa_warped1InverseWarp.nii.gz -o ${DATA_DIR}/prior_estimation/mean_sub.nii.gz

antsApplyTransforms -d 3 -e 3 -i ${PRIOR_DIR}/log_cov.nii.gz -r ${DATA_DIR}/dwi_psc_connectome/diffusion/dti/fa.nii.gz -t [ ${DATA_DIR}/prior_estimation/fa_warped0GenericAffine.mat , 1 ] \
-t ${DATA_DIR}/prior_estimation/fa_warped1InverseWarp.nii.gz -o ${DATA_DIR}/prior_estimation/log_cov_sub.nii.gz
