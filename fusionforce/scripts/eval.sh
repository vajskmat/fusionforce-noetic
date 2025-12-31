#!/bin/bash

echo "Source ROS workspace..."

SEQ=val
BATCH_SIZE=1
TERRAIN_ENCODERS=(lss voxelnet pointpillars bevfusion bevfusion2)
TRAJ_PREDICTORS=(dphysics)
VIS=False

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
    WEIGHTS=/home/robot/Desktop/rci_results/results_withtout_physics/weights/${TERRAIN_ENCODER}/val.pth
    echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
              --terrain_encoder_path ${WEIGHTS} \
              --traj_predictor ${TRAJ_PREDICTOR} \
              --batch_size ${BATCH_SIZE} \
              --seq ${SEQ} \
              --vis ${VIS}
  done
done

echo "Done evaluating."
