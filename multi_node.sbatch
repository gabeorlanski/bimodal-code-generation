#!/bin/bash
#SBATCH --output=./sbatch_logs/%n_test_multi_node.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=cds
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR_JOB=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT_JOB=32500
export WORLD_SIZE=2

echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "MASTER_ADDR_JOB="$MASTER_ADDR_JOB
echo "MASTER_PORT_JOB=$MASTER_PORT_JOB"
echo "$SLURM_NODEID"
srun $(which singularity) exec --nv \
	--overlay $SCRATCH/overlay-50G-10M.ext3:ro \
	/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	/bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
#python scripts/prepare_train_environment.py sample_multinode.yaml -force
echo "Starting"
python -m torch.distributed.launch \
		--nnodes=2 \
      --nproc_per_node=1 \
      --master_addr=$MASTER_ADDR_JOB \
      --master_port=$MASTER_PORT_JOB \
      train.py --debug from_config data/sample_tensorized_cfg.yaml
echo "Finished"
"
