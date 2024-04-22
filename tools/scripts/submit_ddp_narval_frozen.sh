#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2                     # Request 4 GPUs
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1                 # num tasks== num nodes
#SBATCH --time=01:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=24                  # CPU cores/threads
#SBATCH --mem=200G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/waymo_models/pointrcnn_minkunet.yaml
PRETRAINED_MODEL_1=None
PRETRAINED_MODEL_2=None
BATCH_SIZE_PER_GPU=4
TCP_PORT=18888
EXTRA_TAG='default'
NUM_EPOCHS_1=-1
NUM_EPOCHS_2=-1
TEST_START_EPOCH=0


# ========== WAYMO ==========
DATA_DIR_BIND=/home/$USER/scratch/Datasets/Waymo:/OpenPCDet/data/waymo
WAYMO_DATA_DIR=/home/$USER/scratch/Datasets/Waymo
NUSCENES_DATA_DIR=/home/$USER/projects/def-swasland-ab/datasets/nuscenes

SING_IMG=/home/$USER/scratch/singularity/ssl_openpcdet_waymo.sif
TEST_ONLY=false

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
train.py parameters
[--cfg_file CFG_FILE]
[--pretrained_model PRETRAINED_MODEL]
[--tcp_port TCP_PORT]

additional parameters
[--data_dir DATA_DIR_BIND]
[--sing_img SING_IMG]
[--test_only]

--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--pretrained_model     PRETRAINED_MODEL   Pretrained model                    [default=$PRETRAINED_MODEL]
--tcp_port             TCP_PORT           TCP port for distributed training   [default=$TCP_PORT]


--data_dir             DATA_DIR_BIND           Data directory               [default=$DATA_DIR_BIND]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--test_only            TEST_ONLY          Test only flag                      [default=$TEST_ONLY]
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
    -p|--pretrained_model_1)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL_1=$2
            shift
        else
            die 'ERROR: "--pretrained_model 1" requires a non-empty option argument.'
        fi
        ;;
    -f|--pretrained_model_2)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL_2=$2
            shift
        else
            die 'ERROR: "--pretrained_model 2" requires a non-empty option argument.'
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
    -e|--test_start_epoch)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_START_EPOCH=$2
            shift
        else
            die 'ERROR: "--test_start_epoch" requires a non-empty option argument.'
        fi
        ;;
    -a|--num_epochs_1)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_EPOCHS_1=$2
            shift
        else
            die 'ERROR: "--num_epochs_1" requires a non-empty option argument.'
        fi
        ;;
    -d|--num_epochs_2)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_EPOCHS_2=$2
            shift
        else
            die 'ERROR: "--num_epochs_2" requires a non-empty option argument.'
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
NUM_EPOCHS=$NUM_EPOCHS

Additional parameters
DATA_DIR_BIND=$DATA_DIR_BIND
SING_IMG=$SING_IMG
TEST_ONLY=$TEST_ONLY
EXTRA_TAG=$EXTRA_TAG
BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

MASTER_ADDR=$(hostname)


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

TRAIN_CMD_1=$BASE_CMD
TRAIN_CMD_1+="python -m torch.distributed.launch 
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/OpenPCDet/tools/train.py 
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE 
--pretrained_model $PRETRAINED_MODEL_1 
--fix_random_seed
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_frozen_bb"
--epochs $NUM_EPOCHS_1 
--freeze_bb
"


TEST_CMD_1=$BASE_CMD
TEST_CMD_1+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
/OpenPCDet/tools/test.py
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE
--batch_size 12 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_frozen_bb"
--start_epoch $((NUM_EPOCHS_1-10)) 
--eval_all"

TRAIN_CMD_2=$BASE_CMD
TRAIN_CMD_2+="python -m torch.distributed.launch 
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/OpenPCDet/tools/train.py 
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE 
--pretrained_model $PRETRAINED_MODEL_2 
--fix_random_seed
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_2}"
--epochs $NUM_EPOCHS_2 
--load_whole_model
"


TEST_CMD_2=$BASE_CMD
TEST_CMD_2+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
/OpenPCDet/tools/test.py
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE
--batch_size 12 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_2}"
--start_epoch $((NUM_EPOCHS_2-20))
--eval_all"

if [ $TEST_ONLY == "true" ]
then
    echo "Running ONLY evaluation"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TEST_CMD_1"
    eval $TEST_CMD_1
    echo "Done evaluation 1"

    echo "$TEST_CMD_2"
    eval $TEST_CMD_2
    echo "Done evaluation 2"
else
    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD_1"
    eval $TRAIN_CMD_1
    echo "Done training 1"

    echo "$TRAIN_CMD_2"
    eval $TRAIN_CMD_2
    echo "Done training 2"

    echo "Running evaluation"
    echo "$TEST_CMD_1"
    eval $TEST_CMD_1
    echo "Done evaluation 1"

    echo "$TEST_CMD_2"
    eval $TEST_CMD_2
    echo "Done evaluation 2"
fi
