#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=24GB

DATA_DIR=${1}
PRIOR_DIR=${2}
TEMPLATE=${PRIOR_DIR}/template/template.nii.gz
SUBJECT_DIR=${PRIOR_DIR}/training/${3}

# Register geom_field.nii.gz to the template (Move geom_field.nii.gz to template.nii.gz) 
antsRegistration --dimensionality 3 --float 0 \
        --output [${SUBJECT_DIR}/geom_field_output, ${SUBJECT_DIR}/geom_field_outputWarped.nii.gz, ${SUBJECT_DIR}/geom_field_outputInverseWarped.nii.gz] \
        --interpolation Linear --use-histogram-matching 0 \
        --winsorize-image-intensities [0.005,0.995] \
        --initial-moving-transform [${TEMPLATE}, ${SUBJECT_DIR}/geom_field.nii.gz, 1] \
        --transform Rigid['0.2'] \
        --metric MI[${TEMPLATE}, ${SUBJECT_DIR}/geom_field.nii.gz, 1, 32, Regular, 0.25] \
        --convergence [500x250x125x50, 1e-6, 10] --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0 \
        --transform Affine['0.2'] \
        --metric MI[${TEMPLATE}, ${SUBJECT_DIR}/geom_field.nii.gz, 1, 32, Regular, 0.25] \
        --convergence [500x250x125x50, 1e-6, 10] --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0 \
        --transform SyN[0.1, 3, 0] \
        --metric MI[${TEMPLATE}, ${SUBJECT_DIR}/geom_field.nii.gz, 1, 32] \
        --metric CC[${TEMPLATE}, ${SUBJECT_DIR}/geom_field.nii.gz, 1, 4] \
        --convergence [50x25x10, 1e-6, 10] --shrink-factors 4x2x1 \
        --smoothing-sigmas 3x2x1

# Apply the transformation to the signal_tensor.nii.gz
antsApplyTransforms -d 3 -e 3 -i ${SUBJECT_DIR}/signal_tensor.nii.gz -r ${TEMPLATE} \
            -o ${SUBJECT_DIR}/signal_tensor_registered.nii.gz -n Linear \
            -t ${SUBJECT_DIR}/geom_field_output1Warp.nii.gz -t ${SUBJECT_DIR}/geom_field_output0GenericAffine.mat

