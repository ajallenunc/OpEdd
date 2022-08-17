#!/bin/bash
#SBATCH -t 02:00:00 
#SBATCH -c 2 
#SBATCH -n 1 
#SBATCH --mem-per-cpu=20GB

SUBJECT_DIR=${1}

python create_eigenbasis.py --subject_dir $SUBJECT_DIR --sh_order 8

