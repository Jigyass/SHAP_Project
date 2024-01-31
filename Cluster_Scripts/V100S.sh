#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH -mem=32G 
#SBATCH -t 02:00:00
#SBATCH -J VGG16
#SBATCH -o slurm-%j.out


