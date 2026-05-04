#!/bin/bash
#SBATCH --job-name=vit_embeddings
#SBATCH --output=embeddings.out
#SBATCH --error=embeddings.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05.0.1
conda activate /home/hice1/zstern7/scratch/dl_convnet

python -u extract_embeddings.py