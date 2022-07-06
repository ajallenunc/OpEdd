#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --time=100:00
#SBATCH --mem-per-cpu=200GB

PRIOR_DIR=${1}

python build_prior.py --prior_dir $PRIOR_DIR --sh_order 8