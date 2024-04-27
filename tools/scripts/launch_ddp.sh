#!/bin/bash

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))
WORLD_SIZE=$((NUM_GPUS * SLURM_NNODES))
WORKERS_PER_GPU=$(($SLURM_CPUS_PER_TASK / $NUM_GPUS))


echo "NUM GPUS in Node $SLURM_NODEID: $NUM_GPUS"
echo "Node $SLURM_NODEID says: main node at $MASTER_ADDR:$MASTER_PORT"
echo "Node $SLURM_NODEID says: WORLD_SIZE=$WORLD_SIZE"
echo "Node $SLURM_NODEID says: WORKERS_PER_GPU=$SLURM_CPUS_PER_TASK / $NUM_GPUS=$WORKERS_PER_GPU"
echo "Node $SLURM_NODEID says: Loading Singularity Env..."

# Load Singularity
module load StdEnv/2020
module load apptainer #singularity/3.7

PROJ_DIR=$PWD
OPENPCDET_BINDS=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)
    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        OPENPCDET_BINDS+="--bind $entry:/OpenPCDet/pcdet/$name
"
    fi
done

# Extra binds
OPENPCDET_BINDS+="
    --bind $PROJ_DIR/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py:/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_modules.py
    --bind $PROJ_DIR/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py:/OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/pointnet2_utils.py
"

BASE_CMD="APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
APPTAINERENV_WANDB_API_KEY=$WANDB_API_KEY
APPTAINERENV_WANDB_MODE=offline
APPTAINERENV_NCCL_BLOCKING_WAIT=1
apptainer exec
--nv
--pwd /OpenPCDet/tools
--bind $PROJ_DIR/checkpoints:/OpenPCDet/checkpoints
--bind $PROJ_DIR/output:/OpenPCDet/output
--bind $PROJ_DIR/tools:/OpenPCDet/tools
--bind $PROJ_DIR/lib:/OpenPCDet/lib
--bind $DATA_DIR_BIND
$OPENPCDET_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
TRAIN_CMD+="python -m torch.distributed.launch 
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/OpenPCDet/tools/train.py 
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE 
--pretrained_model $PRETRAINED_MODEL 
--fix_random_seed
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag $EXTRA_TAG
--epochs $NUM_EPOCHS
"


TEST_CMD=$BASE_CMD

if [ "$CKPT_TO_EVAL" == 'default' ]; then
    
    TEST_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
    /OpenPCDet/tools/test.py
    --launcher pytorch 
    --cfg_file /OpenPCDet/$CFG_FILE
    --batch_size 12 
    --workers $WORKERS_PER_GPU 
    --extra_tag $EXTRA_TAG
    --test_sample_interval $TEST_SAMPLE_INTERVAL 
    --eval_tag $EVAL_TAG 
    --start_epoch $TEST_START_EPOCH
    --eval_all"

else

    TEST_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
    /OpenPCDet/tools/test.py
    --launcher pytorch 
    --cfg_file /OpenPCDet/$CFG_FILE
    --batch_size 12 
    --workers $WORKERS_PER_GPU 
    --extra_tag $EXTRA_TAG
    --test_sample_interval $TEST_SAMPLE_INTERVAL 
    --eval_tag $EVAL_TAG 
    --ckpt $CKPT_TO_EVAL"

fi

if [ $TEST_ONLY == "true" ]
then
    echo "Running ONLY evaluation"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TEST_CMD"
    eval $TEST_CMD
    echo "Done evaluation"
else
    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done training"

    echo "Running evaluation"
    echo "$TEST_CMD"
    eval $TEST_CMD
    echo "Done evaluation"
fi
