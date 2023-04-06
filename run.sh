#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.

#SBATCH -J 6211_baseline #Slurm job name

# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
#SBATCH --mail-user=kcyipae@connect.ust.hk
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) to use. Note: replace <partition_to_use> with the name of partition
#SBATCH --partition=gpu-share 

# Use 1 nodes and 10 cores
#SBATCH -N 1 -n 8 --gres=gpu:1

# Setup runtime environment if necessary
# For example, setup intel MPI environment
# Go to the job submission directory and run your application
module load anaconda3
module add cuda
source activate 6211
cd $HOME/6211H_Final/code/baseline_model/ChestXray/

srun -n 8 --gres=gpu:1 --partition=gpu-share python train.py 