#!/bin/bash

# ---- SLURM SETTINGS ---- #

# -- Job Specific -- #

#SBATCH --job-name="Scholarly Search"     # What is your job called?

#SBATCH --output=output/main.txt     # Output file - Use %j to inject job id, like output-%j.txt

#SBATCH --error=error/main.txt       # Error file - Use %j to inject job id, like error-%j.txt

#SBATCH --partition=GPU        # Which group of nodes do you want to use? Use "GPU" for graphics card support

#SBATCH --time=7-00:00:00       # What is the max time you expect the job to finish by? DD-HH:MM:SS



# -- Resource Requirements -- #

#SBATCH --mem=64G                # How much memory do you need?

#SBATCH --ntasks-per-node=4     # How many CPU cores do you want to use per node (max 64)?

#SBATCH --nodes=1               # How many nodes do you need to use at once?

#SBATCH --gpus=2               # How many GPUs do you need (max 3)? Remove first "#" to enable.



# -- Email Support -- #

#SBATCH --mail-type=END                # What notifications should be emailed about? (Options: NONE, ALL, BEGIN, END, FAIL, QUEUE)



# ---- YOUR SCRIPT ---- #

python3.9 ./py_code/main.py "test.txt"
