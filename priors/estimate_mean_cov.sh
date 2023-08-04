#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=300GB

PRIOR_DIR=${1}

python build_prior.py --prior_dir $PRIOR_DIR --sh_order 8
