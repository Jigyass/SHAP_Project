#!/bin/bash

#SBATCH -p gpu                   # Specify the GPU partition
#SBATCH --gres="gpu:v100s:1"     # Request 1 Nvidia v100s GPU
#SBATCH -N 1                     # Request 1 node
#SBATCH -n 1                     # Run 1 task
#SBATCH -c 1                     # Request 1 CPU core
#SBATCH --mem=30G                # Request 4GB of memory
#SBATCH -t 01:00:00              # Set maximum runtime to 1 hour
#SBATCH -J clusterJob            # Set the job name to "clusterJob"
#SBATCH -o clusterJob-%j.out     # Set the output file name, %j will be replaced with the job ID

# Load the Python 3.9.12 module
module load Python/3.9.12
module load TensorFlow/2.11-cp310-gpu

# Ensure PIP_TARGET is set for the current session, in case it's not already set by .bashrc for non-interactive sessions
export PIP_TARGET=/scratch/j597s263/.local/pip

# Check if SHAP is already installed in the target directory; if not, install it
if [ ! -d "$PIP_TARGET/shap" ]; then
    pip install shap
else
    echo "SHAP is already installed."
fi

# Run your Python script
python clusterJob.py
