#!/bin/bash

#SBATCH -A eclarson_sm_taylor_0001
#SBATCH --job-name=":3 NIAH :3"
#SBATCH -p batch
####SBATCH --exclusive
#SBATCH -o runjob_niah.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=500G

# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=1


source ~/.bashrc
/users/gmongaras/miniconda3/bin/python GPT_Trainer/niah.py
