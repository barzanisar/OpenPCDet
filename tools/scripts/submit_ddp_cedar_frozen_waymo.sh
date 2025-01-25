#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1                     # Request 1 GPUs
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1                 # num tasks== num nodes
#SBATCH --time=01:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=def-swasland-ab
#SBATCH --cpus-per-task=16                  # CPU cores/threads
#SBATCH --mem=80G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/waymo_models/pointrcnn_minkunet.yaml
PRETRAINED_MODEL_1=None
PRETRAINED_MODEL_2=None
BATCH_SIZE_PER_GPU=8
TEST_BATCH_SIZE_PER_GPU=20
TCP_PORT=18888
EXTRA_TAG='default'
NUM_EPOCHS_1=-1
NUM_EPOCHS_2=-1
TEST_SAMPLE_INTERVAL=-1
TEST_START_EPOCH=0
CKPT_TO_EVAL='default'
CKPT_DIR='default'
EVAL_TAG='default'
WANDB_RUN_NAME=None
WANDB_GROUP=None
LOAD_NUM_BATCHES_TRACKED=false


# ========== WAYMO ==========
DATA_DIR_BIND=/home/$USER/scratch/Datasets/Waymo:/OpenPCDet/data/waymo
WAYMO_DATA_DIR=/home/$USER/scratch/Datasets/Waymo
NUSCENES_DATA_DIR=/home/$USER/projects/def-swasland-ab/datasets/nuscenes
KITTI_DATA_DIR=/home/$USER/projects/def-swasland-ab/datasets3/Kitti

SING_IMG=/home/$USER/scratch/singularity/ssl_openpcdet_waymo.sif
MODE=train_all #train_frozen, train_second, train_all, test_only
EVAL_ALL=false

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
[--mode]

--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--pretrained_model     PRETRAINED_MODEL   Pretrained model                    [default=$PRETRAINED_MODEL]
--tcp_port             TCP_PORT           TCP port for distributed training   [default=$TCP_PORT]


--data_dir             DATA_DIR_BIND           Data directory               [default=$DATA_DIR_BIND]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--mode            MODE          Test only flag                      [default=$MODE]
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
    -a|--cfg_file)       # Takes an option argument; ensure it has been specified.
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
            elif [[ "$CFG_FILE"  == *"kitti_models"* ]]; then
                DATASET=kitti
                echo "KITTI dataset cfg file"
            else
                die 'ERROR: Could not determine backbone from cfg_file path.'
            fi
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -b|--pretrained_model_1)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL_1=$2
            shift
        else
            die 'ERROR: "--pretrained_model 1" requires a non-empty option argument.'
        fi
        ;;
    -c|--pretrained_model_2)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            PRETRAINED_MODEL_2=$2
            shift
        else
            die 'ERROR: "--pretrained_model 2" requires a non-empty option argument.'
        fi
        ;;
    -d|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    -e|--batch_size_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            BATCH_SIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--train_batch_size" requires a non-empty option argument.'
        fi
        ;;
    -f|--mode)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            MODE=$2
            shift
        else
            die 'ERROR: "--mode" requires a non-empty option argument.'
        fi
        ;;
    -g|--extra_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EXTRA_TAG=$2
            shift
        else
            die 'ERROR: "--extra_tag" requires a non-empty option argument.'
        fi
        ;;
    -h|--test_start_epoch)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_START_EPOCH=$2
            shift
        else
            die 'ERROR: "--test_start_epoch" requires a non-empty option argument.'
        fi
        ;;
    -i|--num_epochs_1)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_EPOCHS_1=$2
            shift
        else
            die 'ERROR: "--num_epochs_1" requires a non-empty option argument.'
        fi
        ;;
    -j|--num_epochs_2)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            NUM_EPOCHS_2=$2
            shift
        else
            die 'ERROR: "--num_epochs_2" requires a non-empty option argument.'
        fi
        ;;
    -k|--test_sample_interval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_SAMPLE_INTERVAL=$2
            shift
        else
            die 'ERROR: "--test_sample_interval" requires a non-empty option argument.'
        fi
        ;;
    -l|--ckpt_to_eval)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CKPT_TO_EVAL=$2
            shift
        else
            die 'ERROR: "--ckpt_to_eval" requires a non-empty option argument.'
        fi
        ;;
    -m|--eval_tag)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EVAL_TAG=$2
            shift
        else
            die 'ERROR: "--eval_tag" requires a non-empty option argument.'
        fi
        ;;
    -n|--wandb_run_name)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WANDB_RUN_NAME=$2
            shift
        else
            die 'ERROR: "--wandb_run_name" requires a non-empty option argument.'
        fi
        ;;
    -o|--wandb_group)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WANDB_GROUP=$2
            shift
        else
            die 'ERROR: "--wandb_group" requires a non-empty option argument.'
        fi
        ;;
    -p|--ckpt_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CKPT_DIR=$2
            shift
        else
            die 'ERROR: "--ckpt_dir" requires a non-empty option argument.'
        fi
        ;;
    -q|--load_num_batches_tracked)       # Takes an option argument; ensure it has been specified.
        LOAD_NUM_BATCHES_TRACKED="true"
        ;;
    -r|--test_batch_size_per_gpu)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TEST_BATCH_SIZE_PER_GPU=$2
            shift
        else
            die 'ERROR: "--test_batch_size_per_gpu" requires a non-empty option argument.'
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
******************************************************
CFG_FILE=$CFG_FILE
PRETRAINED_MODEL_1=$PRETRAINED_MODEL_1
PRETRAINED_MODEL_2=$PRETRAINED_MODEL_2
BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU
TEST_BATCH_SIZE_PER_GPU=$TEST_BATCH_SIZE_PER_GPU
TCP_PORT=$TCP_PORT
EXTRA_TAG=$EXTRA_TAG
NUM_EPOCHS_1=$NUM_EPOCHS_1
NUM_EPOCHS_2=$NUM_EPOCHS_2
TEST_SAMPLE_INTERVAL=$TEST_SAMPLE_INTERVAL
TEST_START_EPOCH=$TEST_START_EPOCH
CKPT_TO_EVAL=$CKPT_TO_EVAL
CKPT_DIR=$CKPT_DIR
EVAL_TAG=$EVAL_TAG
WANDB_RUN_NAME=$WANDB_RUN_NAME
WANDB_GROUP=$WANDB_GROUP
LOAD_NUM_BATCHES_TRACKED=$LOAD_NUM_BATCHES_TRACKED


# ========== WAYMO ==========
WAYMO_DATA_DIR=$WAYMO_DATA_DIR
NUSCENES_DATA_DIR=$NUSCENES_DATA_DIR
KITTI_DATA_DIR=$KITTI_DATA_DIR
DATA_DIR_BIND=$DATA_DIR_BIND

SING_IMG=$SING_IMG
MODE=$MODE
EVAL_ALL=$EVAL_ALL
******************************************************
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

if [ "$DATASET" == 'kitti' ]; then
    # Extract Dataset
    echo "Extracting kitti data"

    TMP_DATA_DIR=$SLURM_TMPDIR/data
    unzip -qq $KITTI_DATA_DIR/data_object_calib.zip -d $TMP_DATA_DIR
    unzip -qq $KITTI_DATA_DIR/data_object_label_2.zip -d $TMP_DATA_DIR
    unzip -qq $KITTI_DATA_DIR/data_object_velodyne.zip -d $TMP_DATA_DIR
    unzip -qq $KITTI_DATA_DIR/planes.zip -d $TMP_DATA_DIR
    
    echo "Done extracting kitti data"

    # Extract dataset infos
    echo "Extracting kitti infos"
    unzip -qq $KITTI_DATA_DIR/Infos/kitti_train_infos_5.zip -d $TMP_DATA_DIR
    unzip -qq $KITTI_DATA_DIR/Infos/kitti_infos.zip -d $TMP_DATA_DIR # contains gt database with labels
    echo "Done extracting kitti infos"

    DATA_DIR_BIND=$TMP_DATA_DIR:/OpenPCDet/data/kitti

fi

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
--bind $PROJ_DIR/data/kitti/ImageSets:/OpenPCDet/data/kitti/ImageSets
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
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_frozen_bb"
--epochs $NUM_EPOCHS_1 
--freeze_bb 
--disable_wandb
"

if [ "$LOAD_NUM_BATCHES_TRACKED" == 'true' ]; then
    TRAIN_CMD_1+=" --load_num_batches_tracked"
fi

TEST_CMD_1=$BASE_CMD
TEST_CMD_1+="python -m torch.distributed.launch
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
/OpenPCDet/tools/test.py
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE
--batch_size $TEST_BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_frozen_bb"
--start_epoch $((NUM_EPOCHS_1-10)) 
--eval_all 
--disable_wandb"

TRAIN_CMD_2=$BASE_CMD
TRAIN_CMD_2+="python -m torch.distributed.launch 
--nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
/OpenPCDet/tools/train.py 
--launcher pytorch 
--cfg_file /OpenPCDet/$CFG_FILE 
--pretrained_model $PRETRAINED_MODEL_2 
--batch_size $BATCH_SIZE_PER_GPU 
--workers $WORKERS_PER_GPU 
--extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_ep${NUM_EPOCHS_2}"
--epochs $NUM_EPOCHS_2 
--wandb_run_name $WANDB_RUN_NAME 
--wandb_group $WANDB_GROUP
"

if [ "$NUM_EPOCHS_1" -gt 0 ]; then
  TRAIN_CMD_2+=" --load_whole_model"
fi
 
 
TEST_CMD_2=$BASE_CMD
if [ "$CKPT_TO_EVAL" == 'default' ]; then
    # eval all
    TEST_CMD_2+="python -m torch.distributed.launch
    --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
    /OpenPCDet/tools/test.py
    --launcher pytorch 
    --cfg_file /OpenPCDet/$CFG_FILE
    --batch_size $TEST_BATCH_SIZE_PER_GPU 
    --workers $WORKERS_PER_GPU 
    --start_epoch $TEST_START_EPOCH
    --test_sample_interval $TEST_SAMPLE_INTERVAL 
    --eval_tag $EVAL_TAG 
    --eval_all 
    --wandb_run_name $WANDB_RUN_NAME 
    --wandb_group $WANDB_GROUP"

    if [[ "$EXTRA_TAG" == *"scratch"* ]]; then
        TEST_CMD_2+=" --extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}""
    else
        TEST_CMD_2+=" --extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_ep${NUM_EPOCHS_2}""
    fi

    if [ "$MODE" == "test_only" ]; then
        echo "$TEST_CMD_2"
        eval $TEST_CMD_2
        echo "Done evaluation 2"
    fi

else
    IFS=',' read -r -a CKPT_ARRAY <<< "$CKPT_TO_EVAL"

    # Loop through each ckpt_to_eval list 
    for ckpt in "${CKPT_ARRAY[@]}"; do
        TEST_CMD_2=$BASE_CMD
        TEST_CMD_2+="python -m torch.distributed.launch
        --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0 
        /OpenPCDet/tools/test.py
        --launcher pytorch 
        --cfg_file /OpenPCDet/$CFG_FILE
        --batch_size $TEST_BATCH_SIZE_PER_GPU 
        --workers $WORKERS_PER_GPU 
        --test_sample_interval $TEST_SAMPLE_INTERVAL 
        --eval_tag $EVAL_TAG 
        --ckpt $CKPT_DIR/checkpoint_epoch_"$ckpt".pth
        --disable_wandb"

        if [[ "$EXTRA_TAG" == *"scratch"* ]]; then
            TEST_CMD_2+=" --extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}""
        else
            TEST_CMD_2+=" --extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}_ep${NUM_EPOCHS_2}""
        fi

        if [ "$MODE" == "test_only" ]; then
            echo "$TEST_CMD_2"
            eval $TEST_CMD_2
            echo "Done evaluation for checkpoint_epoch_"$CKPT_TO_EVAL".pth"
        fi
    done
fi

if [ "$MODE" == "scratch" ]; then
    TRAIN_CMD=$BASE_CMD
    TRAIN_CMD+="python -m torch.distributed.launch 
    --nproc_per_node=$NUM_GPUS --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$TCP_PORT --max_restarts=0
    /OpenPCDet/tools/train.py 
    --launcher pytorch 
    --cfg_file /OpenPCDet/$CFG_FILE 
    --batch_size $BATCH_SIZE_PER_GPU 
    --workers $WORKERS_PER_GPU 
    --extra_tag "${EXTRA_TAG}_ep${NUM_EPOCHS_1}"
    --epochs $NUM_EPOCHS_1
    --wandb_run_name $WANDB_RUN_NAME 
    --wandb_group $WANDB_GROUP
    "

    echo "Running training scratch"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD"
    eval $TRAIN_CMD
    echo "Done training scratch"

    echo "Running evaluation scratch"
    echo "$TEST_CMD_2"
    eval $TEST_CMD_2
    echo "Done evaluation scratch"
fi

# if [ "$MODE" == "test_only" ]
# then
#     # echo "Running ONLY evaluation"
#     # echo "Node $SLURM_NODEID says: Launching python script..."

#     # echo "$TEST_CMD_1"
#     # eval $TEST_CMD_1
#     # echo "Done evaluation 1"

#     echo "$TEST_CMD_2"
#     eval $TEST_CMD_2
#     echo "Done evaluation 2"
if [ "$MODE" == "train_frozen" ]
then
    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD_1"
    eval $TRAIN_CMD_1
    echo "Done training 1"

    echo "Running evaluation"
    echo "$TEST_CMD_1"
    eval $TEST_CMD_1
    echo "Done evaluation 1"
elif [ "$MODE" == "train_second" ]
then
    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD_2"
    eval $TRAIN_CMD_2
    echo "Done training 2"

    echo "Running evaluation"
    echo "$TEST_CMD_2"
    eval $TEST_CMD_2
    echo "Done evaluation 2"
elif [ "$MODE" == "train_all" ];
then

    echo "Running training"
    echo "Node $SLURM_NODEID says: Launching python script..."

    echo "$TRAIN_CMD_1"
    eval $TRAIN_CMD_1
    echo "Done training 1"

    echo "$TRAIN_CMD_2"
    eval $TRAIN_CMD_2
    echo "Done training 2"

    # echo "Running evaluation"
    # echo "$TEST_CMD_1"
    # eval $TEST_CMD_1
    # echo "Done evaluation 1"

    echo "$TEST_CMD_2"
    eval $TEST_CMD_2
    echo "Done evaluation 2"
fi
