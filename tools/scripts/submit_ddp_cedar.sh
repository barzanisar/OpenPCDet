#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4                     # Request 4 GPUs
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1                 # num tasks== num nodes
#SBATCH --time=01:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=def-swasland-ab
#SBATCH --cpus-per-task=32                  # CPU cores/threads
#SBATCH --mem=180G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain
#SBATCH --mail-user=barzanisar93@gmail.com

hostname
nvidia-smi

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/waymo_models/pointrcnn_minkunet.yaml
PRETRAINED_MODEL=None
BATCH_SIZE_PER_GPU=4
TCP_PORT=18888
EXTRA_TAG='default'


# ========== WAYMO ==========
DATA_DIR=/home/$USER/scratch/Datasets/Waymo
WAYMO_DATA_DIR=/home/$USER/scratch/Datasets/Waymo
NUSCENES_DATA_DIR=/home/$USER/projects/def-swasland-ab/datasets/nuscenes
DATASET=waymo

SING_IMG=/home/$USER/scratch/singularity/ssl_openpcdet.sif
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
[--data_dir DATA_DIR]
[--sing_img SING_IMG]
[--test_only]

--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--pretrained_model     PRETRAINED_MODEL   Pretrained model                    [default=$PRETRAINED_MODEL]
--tcp_port             TCP_PORT           TCP port for distributed training   [default=$TCP_PORT]


--data_dir             DATA_DIR           Data directory               [default=$DATA_DIR]
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
                DATA_DIR=$WAYMO_DATA_DIR
                echo "Waymo dataset cfg file"
            elif [[ "$CFG_FILE"  == *"nuscenes_models"* ]]; then
                DATASET=nuscenes
                DATA_DIR=$NUSCENES_DATA_DIR
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
DATA_DIR=$DATA_DIR
SING_IMG=$SING_IMG
TEST_ONLY=$TEST_ONLY
EXTRA_TAG=$EXTRA_TAG
BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

export MASTER_ADDR=$(hostname)
export TCP_PORT=$TCP_PORT
export CFG_FILE=$CFG_FILE
export SING_IMG=$SING_IMG
export DATA_DIR=$DATA_DIR
export DATASET=$DATASET
export PRETRAINED_MODEL=$PRETRAINED_MODEL
export BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU
export TEST_ONLY=$TEST_ONLY
export EXTRA_TAG=$EXTRA_TAG

srun tools/scripts/launch_ddp.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $SING_IMG $DATA_DIR