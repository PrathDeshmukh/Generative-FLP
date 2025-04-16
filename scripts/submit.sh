#!/bin/bash
#SBATCH --job-name=tsne
#SBATCH --mem=24gb
#SBATCH --cpus-per-task=8
#SBATCH --tasks=1
#SBATCH --time=01:00:00
#SBATCH -o /home/ppdeshmu/Generative-FLP/data/tsne.out
hostname
source /home/ppdeshmu/miniconda3/etc/profile.d/conda.sh
conda activate flp-rdk-env

python src/tsne_plotter.py --file_paths data/GA_frags.csv data/FORMED.csv data/QM40.csv 
exit 0

