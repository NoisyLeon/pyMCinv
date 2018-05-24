#!/bin/bash
#SBATCH -J MC
#SBATCH -o MC_%j.out
#SBATCH -e MC_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=168:00:00
#SBATCH --mem=MaxMemPerNode

dir=/projects/life9360/code/pyMCinv
cd $dir
python test_h5dbase.py
