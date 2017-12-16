#!/bin/sh
#SBATCH --partition sgpu      # partition requested
#SBATCH -N 6      # nodes requested
#SBATCH -c 2      # cores requested
#SBATCH --output /projects/akar9135/proj_final/test-ouput-lstm_adagrad-3.txt
module load gcc
module load cudnn
module load cuda
module load python/3.5.1
source activate /projects/akar9135/sample/
module load python/3.5.1
python lstm_model_adagrad3.py
source deactivate
