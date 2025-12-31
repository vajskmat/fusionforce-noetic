#!/bin/bash

MODEL=voxelnet  # lss, voxelnet, pointpillars, bevfusion (lss + voxelnet), bevfusion2 (lss + pointpillars)
ROBOT=marv
DEBUG=False
VIS=False
BSZ=24  # 24, 24, 12
WEIGHTS=$HOME/home/fusionforce/fusionforce/config/weights/val.pth # path to pretrained weights

./train.py --bsz $BSZ --nepochs 1000 --lr 1e-4 \
           --debug $DEBUG --vis $VIS \
           --geom_weight 1.0 --terrain_weight 5.0 --phys_weight 5.0 \
           --traj_sim_time 5.0 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
