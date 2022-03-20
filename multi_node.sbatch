#!/bin/bash
#SBATCH --output=./sbatch_logs/test_multi_node.out
#SBATCH --nodes=2
#SBATCH --account=cds
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 16

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR="${master_addr}.hpc.nyu.edu"
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=2

echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT=$MASTER_PORT"

srun $(which singularity) exec --nv \
	--overlay $SCRATCH/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
#python scripts/prepare_train_environment.py sample_multinode.yaml -force
python -m torch.distributed.launch \
      --nproc_per_node=1 \
      --nnodes=$WORLD_SIZE \
      train.py --debug from_config data/sample_tensorized_cfg.yaml
"
