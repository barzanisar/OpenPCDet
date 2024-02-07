#!/bin/bash

# PARAMETERS
# Read symlinks
# KITTI_TRAIN=$(readlink -f ../data/kitti/training)
# KITTI_TEST=$(readlink -f ../data/kitti/testing)
# WAYMO_RAW=$(readlink -f ../data/waymo/raw_data)
# WAYMO_PROCESSED=$(readlink -f ../data/waymo/waymo_processed_data)
# DENSE_LIDAR=$(readlink -f ./data/dense/lidar_hdl64_strongest)
# SNOWFALL_LIDAR=$(readlink -f ./data/dense/snowfall_simulation)
# SNOWFALL_LIDAR_FOV=$(readlink -f ./data/dense/snowfall_simulation_FOV)
# #SNOWFLAKES=$(readlink -f ./data/dense/snowflakes)
# DROR=$(readlink -f ./data/dense/DROR)

# # Infos
# GT_DB=$(readlink -f ./data/dense/gt_database_train_clear_60)
# GT_DB_INFO=$(readlink -f ./data/dense/dense_dbinfos_train_clear_60.pkl)
# TRAIN_INFO=$(readlink -f ./data/dense/dense_infos_train_clear_FOV3000_60.pkl)
# PLANES=$(readlink -f ./data/dense/velodyne_planes)


# Setup volume linking
CUR_DIR=$(pwd)
PROJ_DIR=$CUR_DIR
# KITTI_TRAIN=$KITTI_TRAIN:/PDV/data/kitti/training
# KITTI_TEST=$KITTI_TEST:/PDV/data/kitti/testing
# WAYMO_RAW=$WAYMO_RAW:/PDV/data/waymo/raw_data
# WAYMO_PROCESSED=$WAYMO_PROCESSED:/PDV/data/waymo/waymo_processed_data
# DENSE_LIDAR=$DENSE_LIDAR:/OpenPCDet/data/dense/lidar_hdl64_strongest
# SNOWFALL_LIDAR=$SNOWFALL_LIDAR:/OpenPCDet/data/dense/snowfall_simulation
# SNOWFALL_LIDAR_FOV=$SNOWFALL_LIDAR_FOV:/OpenPCDet/data/dense/snowfall_simulation_FOV
# #SNOWFLAKES=$SNOWFLAKES:/OpenPCDet/data/dense/snowflakes
# DROR=$DROR:/OpenPCDet/data/dense/DROR
# PLANES=$PLANES:/OpenPCDet/data/dense/velodyne_planes

#Infos
# GT_DB=$GT_DB:/OpenPCDet/data/dense/gt_database_train_clear_60
# GT_DB_INFO=$GT_DB_INFO:/OpenPCDet/data/dense/dense_dbinfos_train_clear_60.pkl
# TRAIN_INFO=$TRAIN_INFO:/OpenPCDet/data/dense/dense_infos_train_clear_FOV3000_60.pkl


PCDET_VOLUMES=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/OpenPCDet/pcdet/$name "
    fi
done

docker run -it --env="WANDB_API_KEY=$WANDB_API_KEY" \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$HOME/.XAUTHORITY:/root/.Xauthority:rw" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --hostname="inside-DOCKER" \
        --name="OpenPCDet" \
        --volume $PROJ_DIR/data:/OpenPCDet/data \
        --volume $PROJ_DIR/output:/OpenPCDet/output \
        --volume $PROJ_DIR/tools:/OpenPCDet/tools \
        --volume $PROJ_DIR/lib:/OpenPCDet/lib \
        $PCDET_VOLUMES \
        --rm \
        ssl_openpcdet:minkunet bash

        # --volume $DENSE_LIDAR \
        # --volume $SNOWFALL_LIDAR \
        # --volume $SNOWFALL_LIDAR_FOV \
        # --volume $DROR \
        # --volume $GT_DB \
        # --volume $GT_DB_INFO \
        # --volume $TRAIN_INFO \
        # --volume $PLANES \