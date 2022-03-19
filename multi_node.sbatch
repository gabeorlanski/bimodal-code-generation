#!/bin/bash
#SBATCH --output=./sbatch_logs/%x_train.out
#SBATCH --nodes=2
#SBATCH --account=cds
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 16

export WORLD_SIZE=2
echo "NODELIST="${SLURM_NODELIST}
echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="${master_addr}"
echo "MASTER_ADDR="$MASTER_ADDR
echo "${SLURM_NODEID}"
MASTER_PORT=$(shuf -i 10000-65500 -n 1)

srun $(which singularity) exec --nv \
	--overlay $SCRATCH/overlay-50G-10M.ext3:rw \
	/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	/bin/bash -c "
source /ext3/env.sh
echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")" >> /etc/hosts
conda activate adversarial-code
#python scripts/prepare_train_environment.py $1 -force
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${master_addr}:${MASTER_PORT}" \
	--node_rank=$SLURM_NODEID \
    train.py --debug from_config data/sample_tensorized_cfg.yaml
"
