#!/bin/bash

#SBATCH -A eclarson_sm_taylor_0001
#SBATCH --job-name="8192 squared >w<_Gated_Softmax"
#SBATCH -p batch
####SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --exclude=bcm-dgxa100-0012

seq_len="8192"
run_name="slimpj_medium_8192sl_gpu_32bs__squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2"
attention_type="squared__sm_norm__A_mask_type_neg_softplus__in_conv_k_2"


# Specify node to run on
###SBATCH --nodelist=bcm-dgxa100-0003


# Number of nodes
nnodes=2
# Number of tasks per node
nproc_per_node=8



# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1200  # Increase timeout to 20 minutes

# # Debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1

# Optimizations
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=600


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

source ~/.bashrc
# CUDA_VISIBLE_DEVICES=0,1 
srun /home/gmongaras/miniconda3/bin/torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:30154 \
GPT_Trainer/train.py $seq_len $run_name $attention_type
