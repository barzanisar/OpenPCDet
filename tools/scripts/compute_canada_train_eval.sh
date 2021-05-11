#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=OpenPCDet-train
#SBATCH --account=rrg-swasland
#SBATCH --cpus-per-task=8              # CPU cores/threads
#SBATCH --gres=gpu:t4:2                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-4%1   # 4 is the number of jobs in the chain

# Default Command line args
DATA_DIR=/home/$USER/projects/rrg-swasland/Datasets/Kitti
INFOS_DIR=data/kitti
SING_IMG=/home/$USER/projects/rrg-swasland/singularity/openpcdet.sif
CFG_FILE=cfgs/kitti_models/CaDDN.yaml
TRAIN_BATCH_SIZE=2
TEST_BATCH_SIZE=2
EPOCHS=80
EXTRA_TAG='default'
DIST=true
TCP_PORT=18888
FIX_RANDOM_SEED=false

# Usage info
show_help() {
echo "
Usage: sbatch --job-name=JOB_NAME --mail-user=MAIL_USER --gres=gpu:GPU_ID:NUM_GPUS tools/scripts/${0##*/} [-h]
[--data_dir DATA_DIR]
[--infos_dir INFOS_DIR]
[--sing_img SING_IMG]
[--train_batch_size TRAIN_BATCH_SIZE]
[--test_batch_size TEST_BATCH_SIZE]
[--cfg_file CFG_FILE]
[--epochs EPOCHS]
[--extra_tag 'EXTRA_TAG']
[--dist]
[--tcp_port TCP_PORT]

--data_dir             DATA_DIR           Zipped data directory               [default=$DATA_DIR]
--infos_dir            INFOS_DIR          Infos directory                     [default=$INFOS_DIR]
--sing_img             SING_IMG           Singularity image file              [default=$SING_IMG]
--train_batch_size     TRAIN_BATCH_SIZE   Train batch size                    [default=$TRAIN_BATCH_SIZE]
--test_batch_size      TEST_BATCH_SIZE    Test batch size                     [default=$TEST_BATCH_SIZE]
--cfg_file             CFG_FILE           Config file                         [default=$CFG_FILE]
--epochs               EPOCHS             Epochs                              [default=$EPOCHS]
--extra_tag            EXTRA_TAG          Extra experiment tag                [default=$EXTRA_TAG]
--dist                 DIST               Distributed training flag           [default=$DIST]
--tcp_port             TCP_PORT           TCP port for distributed training   [default=$TCP_PORT]
--fix_random_seed      FIX_RANDOM_SEED    Flag to fix random seed             [default=$FIX_RANDOM_SEED]
"
}

# Get command line arguments
while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -d|--data_dir)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            DATA_DIR=$2
            shift
        else
            die 'ERROR: "--data_dir" requires a non-empty option argument.'
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
    -c|--cfg_file)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            CFG_FILE=$2
            shift
        else
            die 'ERROR: "--cfg_file" requires a non-empty option argument.'
        fi
        ;;
    -e|--epochs)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            EPOCHS=$2
            shift
        else
            die 'ERROR: "--epochs" requires a non-empty option argument.'
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
    -2|--dist)       # Takes an option argument; ensure it has been specified.
        DIST="true"
        ;;
    -o|--tcp_port)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TCP_PORT=$2
            shift
        else
            die 'ERROR: "--tcp_port" requires a non-empty option argument.'
        fi
        ;;
    -f|--fix_random_seed)       # Takes an option argument; ensure it has been specified.
        FIX_RANDOM_SEED="true"
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
DATA_DIR=$DATA_DIR
INFOS_DIR=$INFOS_DIR
SING_IMG=$SING_IMG
CFG_FILE=$CFG_FILE
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE
TEST_BATCH_SIZE=$TEST_BATCH_SIZE
EPOCHS=$EPOCHS
EXTRA_TAG=$EXTRA_TAG
DIST=$DIST
TCP_PORT=$TCP_PORT
FIX_RANDOM_SEED=$FIX_RANDOM_SEED
"

echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

# Extract Dataset
echo "Extracting data"
TMP_DATA_DIR=$SLURM_TMPDIR/data
for file in $DATA_DIR/*.zip; do
    echo "Unziping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting data"

# Extract dataset infos
echo "Extracting dataset infos"
for file in $INFOS_DIR/*.zip; do
    echo "Unziping $file to $TMP_DATA_DIR"
    unzip -qq $file -d $TMP_DATA_DIR
done
echo "Done extracting dataset infos"

module load StdEnv/2020
module load singularity/3.6

PROJ_DIR=$PWD
PCDET_BINDS=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)
    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_BINDS+="--bind $entry:/OpenPCDet/pcdet/$name
"
    fi
done


BASE_CMD="SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1 singularity exec
--nv
--pwd /OpenPCDet/tools
--cleanenv
--bind $PROJ_DIR/checkpoints:/OpenPCDet/checkpoints
--bind $PROJ_DIR/output:/OpenPCDet/output
--bind $PROJ_DIR/tools:/OpenPCDet/tools
--bind $TMP_DATA_DIR:/OpenPCDet/$INFOS_DIR
$PCDET_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD
if [ $DIST != "true" ]
then
    TRAIN_CMD+="python /OpenPCDet/tools/train.py
"
else
    TRAIN_CMD+="python -m torch.distributed.launch
    --nproc_per_node=2
    /OpenPCDet/tools/train.py
    --launcher pytorch
    --sync_bn
    --tcp_port $TCP_PORT"
fi
TRAIN_CMD+="
    --cfg_file $CFG_FILE
    --workers $SLURM_CPUS_PER_TASK
    --batch_size $TRAIN_BATCH_SIZE
    --epochs $EPOCHS
    --extra_tag $EXTRA_TAG
"

if [ $FIX_RANDOM_SEED = "true" ]
then
    TRAIN_CMD+="    --fix_random_seed
"
fi

echo "Running training"
echo "$TRAIN_CMD"
eval $TRAIN_CMD

echo "Done training
"

TEST_CMD=$BASE_CMD
TEST_CMD+="python /OpenPCDet/tools/test.py
    --cfg_file $CFG_FILE
    --workers $SLURM_CPUS_PER_TASK
    --batch_size $TEST_BATCH_SIZE
    --extra_tag $EXTRA_TAG
    --eval_all
"

echo "Running evaluation"
echo "$TEST_CMD"
eval $TEST_CMD

echo "Done evaluation"
