#!/bin/bash



# ---- SLURM SETTINGS ---- #



# -- Job Specific -- #

#SBATCH --job-name="Download_Llama-3-70B_weights"     # What is your job called?

#SBATCH --output=output/dload_llama370B_weights.txt     # Output file - Use %j to inject job id, like output-%j.txt

#SBATCH --error=error/dload_llama370B_weights.txt       # Error file - Use %j to inject job id, like error-%j.txt



#SBATCH --partition=week        # Which group of nodes do you want to use? Use "GPU" for graphics card support

#SBATCH --time=7-00:00:00       # What is the max time you expect the job to finish by? DD-HH:MM:SS



# -- Resource Requirements -- #

#SBATCH --mem=10G                # How much memory do you need?

#SBATCH --ntasks-per-node=4     # How many CPU cores do you want to use per node (max 64)?

#SBATCH --nodes=1               # How many nodes do you need to use at once?

##SBATCH --gpus=1               # How many GPUs do you need (max 3)? Remove first "#" to enable.



# -- Email Support -- #

#SBATCH --mail-type=END                # What notifications should be emailed about? (Options: NONE, ALL, BEGIN, END, FAIL, QUEUE)
huggingface-cli login --token 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

huggingface-cli download meta-llama/Meta-Llama-3.1-70B --include "original/*" --local-dir weights/Meta-Llama-3.1-70B


