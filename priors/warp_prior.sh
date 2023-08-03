#!/bin/bash
#SBATCH -t 02:00:00 
#SBATCH -c 2 
#SBATCH -n 1 
#SBATCH --mem-per-cpu=50GB

PRIOR_DIR=${1}
SUBJECT_DIR=${2}

# Move Template Image (template.nii.gz) to Subject Space (geom_field.nii.gz)
antsRegistrationSyN.sh -d 3 -m ${PRIOR_DIR}/template/template.nii.gz -f ${SUBJECT_DIR}/geom_field.nii.gz \
                                        -o ${SUBJECT_DIR}/geom_field_warped

# Apply Transformations to Mean and Cov Signals 
antsApplyTransforms -d 3 -e 3 -i ${PRIOR_DIR}/template/mean_signal.nii.gz \
-r ${SUBJECT_DIR}/geom_field.nii.gz -o ${SUBJECT_DIR}/mean_func_warped.nii.gz \
-t ${SUBJECT_DIR}/geom_field_warped1Warp.nii.gz ${SUBJECT_DIR}/geom_field_warped0GenericAffine.mat

antsApplyTransforms -d 3 -e 3 -i ${PRIOR_DIR}/template/log_cov_signal.nii.gz \
-r ${SUBJECT_DIR}/geom_field.nii.gz -o ${SUBJECT_DIR}/log_cov_func_warped.nii.gz \
-t ${SUBJECT_DIR}/geom_field_warped1Warp.nii.gz ${SUBJECT_DIR}/geom_field_warped0GenericAffine.mat

