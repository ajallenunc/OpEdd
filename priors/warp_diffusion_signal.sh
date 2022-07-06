#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=100:00
#SBATCH --mem-per-cpu=24GB

DATA_DIR=${1}
PRIOR_DIR=${2}
TEMPLATE=${PRIOR_DIR}/template/template.nii.gz
SUBJECT_DIR=${PRIOR_DIR}/training/${3}

antsRegistrationSyN.sh -d 3 -f ${TEMPLATE} -m ${SUBJECT_DIR}/geom_field.nii.gz \
										-o ${SUBJECT_DIR}/geom_field_warped

antsApplyTransforms -d 3 -e 3 -r $TEMPLATE -i ${SUBJECT_DIR}/signal_tensor.nii.gz \
							-t ${SUBJECT_DIR}/geom_field_warped0GenericAffine.mat \
							-t ${SUBJECT_DIR}/geom_field_warped1Warp.nii.gz \
							-o ${SUBJECT_DIR}/signal_tensor_registered.nii.gz