#!/bin/bash -l

#$ -N HIST             # Job name
#$ -P textconv                  # Project name
#$ -o HIST             # Output file name
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=6
#$ -m ea                        # Email on end or abort
#$ -j y                         # Merge output and error file
#$ -l h_rt=30:00:00

module load miniconda/4.7.5
conda activate FER
export PYTHONPATH=projectnb/textconv/ykh/fer/ensemble/:$PYTHONPATH

python train.py network=vgg name=HIST n_epochs=300
