#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8             # CPU cores/threads
#SBATCH --gres=gpu:t4:4                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-2%1   # 4 is the number of jobs in the chain
#SBATCH --mail-user=barzanisar93@gmail.com

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/kitti_models/pv_rcnn.yaml
TRAIN_BATCH_SIZE='default'
TEST_BATCH_SIZE='default'
WORKERS=$SLURM_CPUS_PER_TASK
EXTRA_TAG='default'
TEST_INFO_PKL='default' # Test only 
EVAL_TAG='default' # Test only 
CKPT=None
PRETRAINED_MODEL=None
TCP_PORT=18888
SYNC_BN=true
FIX_RANDOM_SEED=false
#Save last 20 ckpts with interval 1
CKPT_SAVE_INTERVAL=1
MAX_CKPT_SAVE_NUM=80
SAVE_CKPT_AFTER_EPOCH=0
# Epoch is evaluated if epoch > start epoch = 160 - NUM_EPOCHS_TO_EVAL
NUM_EPOCHS_TO_EVAL=0

# ========== DENSE ==========
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/Dense
INFOS_DIR=/home/$USER/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos

# Additional parameters
DATASET=dense
SING_IMG=/home/$USER/projects/rrg-swasland/singularity/openpcdet_martin.sif
DIST=true
TEST_ONLY=false
WANDB_API_KEY=$WANDB_API_KEY
WANDB_MODE='offline'

# Get last element in string and increment by 1
NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
NUM_GPUS=$(($NUM_GPUS + 1))

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
train.py parameters
[--cfg_file CFG_FILE]
[--train_batch_size TRAIN_BATCH_SIZE]
[--test_batch_size TEST_BATCH_SIZE]
[--extra_tag 'EXTRA_TAG']
[--ckpt CKPT]
[--pretrained_model PRETRAINED_MODEL]
[--tcp_port TCP_PORT]
[--sync_bn SYNC_BN]
[--fix_random_seed]
[--ckpt_save_interval CKPT_SAVE_INTERVAL]
[--max_ckpt_save_num MAX_CKPT_SAVE_NUM]

additional parameters
[--data_dir DATA_DIR]
[--infos_dir INFOS_DIR]
[--sing_img SING_IMG]
[--dist]
[--test_only]

--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--train_batch_size     TRAIN_BATCH_SIZE   Train batch size                    [default=$TRAIN_BATCH_SIZE]
--test_batch_size      TEST_BATCH_SIZE    Test batch size                     [default=$TEST_BATCH_SIZE]
--extra_tag            EXTRA_TAG          Extra experiment tag                [default=$EXTRA_TAG]
--ckpt                 CKPT               Checkpoint to start from            [default=$CKPT]
--pretrained_model     PRETRAINED_MODEL   Pretrained model                    [default=$PRETRAINED_MODEL]
--tcp_port             TCP_PORT           TCP port for distributed training   [default=$TCP_PORT]
--sync_bn              SYNC_BN            Use sync bn                         [default=$SYNC_BN]
--fix_random_seed      FIX_RANDOM_SEED    Flag to fix random seed             [default=$FIX_RANDOM_SEED]
--ckpt_save_interval   CKPT_SAVE_INTERVAL Interval of saving checkpoints      [default=$CKPT_SAVE_INTERVAL]
--save_ckpt_after_epoch  SAVE_CKPT_AFTER_EPOCH Save checkpoint after epoch    [default=$SAVE_CKPT_AFTER_EPOCH]
--max_ckpt_save_num    MAX_CKPT_SAVE_NUM  Max number of saved checkpoints     [default=$MAX_CKPT_SAVE_NUM]
--num_epochs_to_eval    NUM_EPOCHS_TO_EVAL Num last saved ckpts to eval       [default=$NUM_EPOCHS_TO_EVAL]


--data_dir             DATA_DIR           Zipped data directory               [default=$DATA_DIR]
--infos_dir            INFOS_DIR          Infos directory                     [default=$INFOS_DIR]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--dist                 DIST               Distributed training flag           [default=$DIST]
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
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -b|--train_batch_size)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TRAIN_BATCH_SIZE=$2
            shift
        else
            die 'ERROR: "--train_batch_size" requires a non-empty option argument.'
        fi
        ;;
    -a|--test_batch_size)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_BATCH_SIZE=$2
            shift
        else
            die 'ERROR: "--test_batch_size" requires a non-empty option argument.'
        fi
        ;;
    -w|--workers)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WORKERS=$2
            shift
        else
            die 'ERROR: "--workers" requires a non-empty option argument.'
        fi
        ;;
    -t|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -c|--ckpt)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            ckpt=$2
            shift
        else
            die 'ERROR: "--ckpt" requires a non-empty option argument.'
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
    -y|--sync_bn)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SYNC_BN=$2
            shift
        else
            die 'ERROR: "--sync_bn" requires a non-empty option argument.'
        fi
        ;;
    -f|--fix_random_seed)       # Takes an option argument; ensure it has been specified.
        FIX_RANDOM_SEED="true"
        ;;
    -v|--ckpt_save_interval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CKPT_SAVE_INTERVAL=$2
            shift
        else
            die 'ERROR: "--ckpt_save_interval" requires a non-empty option argument.'
        fi
        ;;
    -j|--save_ckpt_after_epoch)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SAVE_CKPT_AFTER_EPOCH=$2
            shift
        else
            die 'ERROR: "--save_ckpt_after_epoch" requires a non-empty option argument.'
        fi
        ;;
    -x|--num_epochs_to_eval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_EPOCHS_TO_EVAL=$2
            shift
        else
            die 'ERROR: "--num_epochs_to_eval" requires a non-empty option argument.'
        fi
        ;;
    -m|--max_ckpt_save_num)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MAX_CKPT_SAVE_NUM=$2
            shift
        else
            die 'ERROR: "--max_ckpt_save_num" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -e|--test_info_pkl)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_INFO_PKL=$2
            shift
        else
            die 'ERROR: "--test_info_pkl" requires a non-empty option argument.'
        fi
        ;;
    # Additional parameters
    -g|--eval_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EVAL_TAG=$2
            shift
        else
            die 'ERROR: "--eval_tag" requires a non-empty option argument.'
        fi
        ;;
    -i|--infos_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            INFOS_DIR=$2
            shift
        else
            die 'ERROR: "--infos_dir" requires a non-empty option argument.'
        fi
        ;;
    -s|--sing_img)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SING_IMG=$2
            shift
        else
            die 'ERROR: "--sing_img" requires a non-empty option argument.'
        fi
        ;;
    -2|--dist)       # Takes an option argument; ensure it has been specified.
        DIST="true"
        ;;
    -z|--test_only)       # Takes an option argument; ensure it has been specified.
        TEST_ONLY="true"
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
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
TEST_BATCH_SIZE=$TEST_BATCH_SIZE
WORKERS=$WORKERS
EXTRA_TAG=$EXTRA_TAG
CKPT=$CKPT
PRETRAINED_MODEL=$PRETRAINED_MODEL
TCP_PORT=$TCP_PORT
SYNC_BN=$SYNC_BN
FIX_RANDOM_SEED=$FIX_RANDOM_SEED
CKPT_SAVE_INTERVAL=$CKPT_SAVE_INTERVAL
SAVE_CKPT_AFTER_EPOCH=$SAVE_CKPT_AFTER_EPOCH
MAX_CKPT_SAVE_NUM=$MAX_CKPT_SAVE_NUM
NUM_EPOCHS_TO_EVAL=$NUM_EPOCHS_TO_EVAL

Additional parameters
DATA_DIR=$DATA_DIR
INFOS_DIR=$INFOS_DIR
SING_IMG=$SING_IMG
DIST=$DIST
TEST_ONLY=$TEST_ONLY
NUM_GPUS=$NUM_GPUS
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""


# Extract Dense Dataset
echo "Extracting Dense data"
TMP_DATA_DIR=$SLURM_TMPDIR/data

echo "Unzipping $DATA_DIR/lidar_hdl64_strongest.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/lidar_hdl64_strongest.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/velodyne_planes.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/velodyne_planes.zip -d $TMP_DATA_DIR

echo "Unzipping $DATA_DIR/DROR.zip to $TMP_DATA_DIR"
unzip -qq $DATA_DIR/DROR.zip -d $TMP_DATA_DIR

echo "Unzipping $INFOS_DIR/infos.zip to $TMP_DATA_DIR"
unzip -qq $INFOS_DIR/infos.zip -d $TMP_DATA_DIR


echo "Done extracting Dense data"


# Load Singularity
module load StdEnv/2020
module load singularity/3.7

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
SINGULARITYENV_WANDB_MODE=$WANDB_MODE
singularity exec
--nv
--pwd /OpenPCDet/tools
--bind $PROJ_DIR/checkpoints:/OpenPCDet/checkpoints
--bind $PROJ_DIR/output:/OpenPCDet/output
--bind $PROJ_DIR/tests:/OpenPCDet/tests
--bind $PROJ_DIR/tools:/OpenPCDet/tools
--bind $PROJ_DIR/lib:/OpenPCDet/lib
--bind $TMP_DATA_DIR:/OpenPCDet/data/$DATASET
--bind $PROJ_DIR/data/$DATASET/ImageSets:/OpenPCDet/data/$DATASET/ImageSets
$OPENPCDET_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
if [ $DIST != "true" ]
then
    TRAIN_CMD+="python /OpenPCDet/tools/train.py
"
else
    TRAIN_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS
    /OpenPCDet/tools/train.py
    --launcher pytorch
    --sync_bn
    --tcp_port $TCP_PORT"
fi
TRAIN_CMD+="
    --cfg_file /OpenPCDet/$CFG_FILE
    --workers $WORKERS
    --pretrained_model $PRETRAINED_MODEL
    --extra_tag $EXTRA_TAG
    --ckpt_save_interval $CKPT_SAVE_INTERVAL
    --save_ckpt_after_epoch $SAVE_CKPT_AFTER_EPOCH
    --max_ckpt_save_num $MAX_CKPT_SAVE_NUM
    --num_epochs_to_eval $NUM_EPOCHS_TO_EVAL
"

# Additional arguments if necessary
if [ $TRAIN_BATCH_SIZE != "default" ]
then
    TRAIN_CMD+="    --batch_size $TRAIN_BATCH_SIZE
"
fi

if [ $FIX_RANDOM_SEED = "true" ]
then
    TRAIN_CMD+="    --fix_random_seed
"
fi

TEST_CMD=$BASE_CMD
if [ $DIST != "true" ]
then
    TEST_CMD+="python /OpenPCDet/tools/test.py
"
else
    TEST_CMD+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS
    /OpenPCDet/tools/test.py
    --launcher pytorch
    --tcp_port $TCP_PORT"
fi
TEST_CMD+="
    --cfg_file /OpenPCDet/$CFG_FILE
    --workers $WORKERS
    --extra_tag $EXTRA_TAG
    --eval_all
"

# Additional arguments if necessary
if [ $TEST_BATCH_SIZE != "default" ]
then
    TEST_CMD+=" --batch_size $TEST_BATCH_SIZE
"
fi

# Additional arguments if necessary
if [ $EVAL_TAG != "default" ]
then
    TEST_CMD+=" --eval_tag $EVAL_TAG
"
fi

# Additional arguments if necessary 
# DONT add any args after --set
if [ $TEST_INFO_PKL != "default" ]
then
    TEST_CMD+=" --set DATA_CONFIG.INFO_PATH.test $TEST_INFO_PKL
"
fi

if [ $TEST_ONLY == "true" ]
then
    echo "Running ONLY evaluation"
    echo "$TEST_CMD"
    eval $TEST_CMD
    echo "Done evaluation"
else
    echo "Running training and evaluation"
    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done training and evaluation"
fi
