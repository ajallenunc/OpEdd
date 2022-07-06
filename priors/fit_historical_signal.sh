#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=100:00
#SBATCH --mem-per-cpu=12GB

DATA_DIR=${1}
PRIOR_DIR=${2}
SUBJECT_ID=${3}

python fit_historical_signal.py --data_dir $DATA_DIR --prior_dir $PRIOR_DIR --subject_id $SUBJECT_ID \
										--mask mask.nii.gz --Bval 2000 --sh_order 8