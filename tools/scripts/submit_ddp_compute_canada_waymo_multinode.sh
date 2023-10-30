#!/bin/bash
#SBATCH --wait-all-nodes=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:4                     # Request 4 GPUs
#SBATCH --ntasks=2 
#SBATCH --ntasks-per-node=1                 # num tasks== num nodes
#SBATCH --time=01:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=16                  # CPU cores/threads
#SBATCH --mem=400G                        # memory per node
#SBATCH --output=./output/log/%x-%j.out     # STDOUT
#SBATCH --array=1-3%1                       # 3 is the number of jobs in the chain
#SBATCH --mail-user=barzanisar93@gmail.com

# die function
die() { echo "$*" 1>&2 ; exit 1; }

# Default Command line args
# train.py script parameters
CFG_FILE=tools/cfgs/waymo_models/centerpoint.yaml
# WORKERS=4 #num workers per gpu so 16 workers per node/4gpus i.e. $SLURM_CPUS_PER_TASK/num gpus per node
EXTRA_TAG='default'
TEST_INFO_PKL='default' # Test only 
EVAL_TAG='default' # Test only 
PRETRAINED_MODEL=None
WORKERS=4
BATCH_SIZE_PER_GPU='default'
TCP_PORT=18888

# ========== WAYMO ==========
DATA_DIR=/home/$USER/scratch/Datasets/Waymo
SING_IMG=/home/$USER/scratch/singularity/openpcdet_martin.sif
TEST_ONLY=false
# WANDB_API_KEY=$WANDB_API_KEY

# # Get last element in string and increment by 1
# NUM_GPUS="${CUDA_VISIBLE_DEVICES: -1}"
# NUM_GPUS=$(($NUM_GPUS + 1))

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
train.py parameters
[--cfg_file CFG_FILE]
[--extra_tag 'EXTRA_TAG']
[--pretrained_model PRETRAINED_MODEL]
[--tcp_port TCP_PORT]

additional parameters
[--data_dir DATA_DIR]
[--sing_img SING_IMG]
[--test_only]

--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--extra_tag            EXTRA_TAG          Extra experiment tag                [default=$EXTRA_TAG]
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
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
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
    -w|--workers)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            WORKERS=$2
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
EXTRA_TAG=$EXTRA_TAG
PRETRAINED_MODEL=$PRETRAINED_MODEL
TCP_PORT=$TCP_PORT

Additional parameters
DATA_DIR=$DATA_DIR
SING_IMG=$SING_IMG
TEST_ONLY=$TEST_ONLY
NUM_GPUS=$NUM_GPUS
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
export PRETRAINED_MODEL=$PRETRAINED_MODEL
export TEST_INFO_PKL=$TEST_INFO_PKL
export TEST_ONLY=$TEST_ONLY
export EXTRA_TAG=$EXTRA_TAG
export EVAL_TAG=$EVAL_TAG
export WORKERS=$WORKERS
export BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU

srun tools/scripts/launch_ddp_waymo.sh #$MASTER_ADDR $TCP_PORT $CFG_FILE $SING_IMG $DATA_DIR