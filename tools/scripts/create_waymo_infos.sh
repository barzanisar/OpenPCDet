#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=OpenPCDet-create_gtdatabase
#SBATCH --account=def-swasland-ab
#SBATCH --cpus-per-task=20             # CPU cores/threads
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --mem=64000M                   # memory per node
#SBATCH --output=./output/log/%x-%j.out   # STDOUT
#SBATCH --mail-type=ALL
#SBATCH --array=1-1%1   
#SBATCH --mail-user=barzanisar93@gmail.com

# die function
die() { echo "$*" 1>&2 ; exit 1; }


# ========== WAYMO ==========
DATA_DIR=/home/$USER/scratch/Datasets/Waymo
SING_IMG=/home/$USER/scratch/singularity/openpcdet_martin.sif

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
--bind $PROJ_DIR/tools:/OpenPCDet/tools
--bind $PROJ_DIR/lib:/OpenPCDet/lib
--bind $DATA_DIR:/OpenPCDet/data/waymo
--bind $PROJ_DIR/data/waymo/ImageSets:/OpenPCDet/data/waymo/ImageSets
$OPENPCDET_BINDS
$SING_IMG
"

TRAIN_CMD=$BASE_CMD

TRAIN_CMD+="python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_gt_database --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml"

echo "Running create infos"
echo "$TRAIN_CMD"
eval $TRAIN_CMD
echo "Done !"

