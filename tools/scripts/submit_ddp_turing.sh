#!/bin/bash

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/waymo_models/pointrcnn_minkunet.yaml
PRETRAINED_MODEL=None
BATCH_SIZE_PER_GPU=4
TCP_PORT=18888
EXTRA_TAG='default'
TEST_ONLY=false


# ========== WAYMO ==========
DATASET=waymo
DATA_DIR_BIND=/raid/datasets/Waymo:/OpenPCDet/data/waymo
WAYMO_DATA_DIR=/raid/datasets/Waymo
KITTI_DATA_DIR=/raid/datasets/semantic_kitti
NUSCENES_DATA_DIR=/raid/datasets/nuscenes
SING_IMG=/raid/home/nisarbar/singularity/ssl_openpcdet_waymo.sif
NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1
MASTER_ADDR=$CLUSTER_NAME
WORKERS_PER_GPU=8 # Turing has 48 cpus so use 10 cpus/gpu

# Usage info
show_help() {
echo "
Usage
"
}

# Change default data_dir and infos_dir for different datasets

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    # train.py parameters
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2

            # Get dataset
            echo "Checking dataset"
            if [[ "$CFG_FILE"  == *"waymo_models"* ]]; then
                DATASET=waymo
                DATA_DIR_BIND=$WAYMO_DATA_DIR:/OpenPCDet/data/waymo
                echo "Waymo dataset cfg file"
            elif [[ "$CFG_FILE"  == *"nuscenes_models"* ]]; then
                DATASET=nuscenes
                DATA_DIR_BIND=$NUSCENES_DATA_DIR:/OpenPCDet/data/nuscenes/v1.0-trainval
                echo "Nuscenes dataset cfg file"
            else
                die 'ERROR: Could not determine backbone from cfg_file path.'
            fi
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -p|--pretrained_model)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL=$2
            shift
        else
            die 'ERROR: "--pretrained_model" requires a non-empty option argument.'
        fi
        ;;
    -o|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    -b|--batch_size_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BATCH_SIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--train_batch_size" requires a non-empty option argument.'
        fi
        ;;
    -z|--test_only)       # Takes an option argument; ensure it has been specified.
        TEST_ONLY="true"
        ;;
    -t|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -n|--num_gpus)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_GPUS=$2
            shift
        else
            die 'ERROR: "--num_gpus" requires a non-empty option argument.'
        fi
        ;;
    -c|--cuda_visible_devices)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CUDA_VISIBLE_DEVICES=$2
            shift
        else
            die 'ERROR: "--cuda_visible_devices" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

echo "Running with the following arguments:
train.py parameters:
CFG_FILE=$CFG_FILE
PRETRAINED_MODEL=$PRETRAINED_MODEL
TCP_PORT=$TCP_PORT

Additional parameters
DATA_DIR=$DATA_DIR_BIND
SING_IMG=$SING_IMG
TEST_ONLY=$TEST_ONLY
EXTRA_TAG=$EXTRA_TAG
BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU
"


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

BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
SINGULARITYENV_NCCL_BLOCKING_WAIT=1
singularity exec
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
"


TEST_CMD=$BASE_CMD

TEST_CMD+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
/OpenPCDet/tools/test.py
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag $EXTRA_TAG
--eval_all"

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
