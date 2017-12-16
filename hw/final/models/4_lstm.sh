#!/bin/sh
#SBATCH --partition sgpu      # partition requested
#SBATCH -N 2      # nodes requested
#SBATCH -c 2      # cores requested
#SBATCH --output /projects/akar9135/proj_final/test-ouput-lstm.txt
module load gcc
module load cudnn
module load cuda
module load python/3.5.1
source activate /projects/akar9135/sample/
module load python/3.5.1
python lstm_model.py
source deactivate
