#!/bin/bash

# Get file path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build and run the docker for deployment
docker build -t rtm_pose_deploy:latest -f $DIR/Dockerfile.deploy $DIR


# Define models to deploy
MODELS="rtm_body8_26keypoints_det-m_pose-m_256x192 rtm_coco_det-m_pose-l rtm_body8_26keypoints_det-m_pose-m_384x288 rtm_coco_det-nano_pose-m rtm_body8_det-m_pose-s rtm_body8_det-m_pose-m"

# Deploy each model
for model in $MODELS; do
    docker run --name rtm_pose_deploy_$model \
        --env-file $DIR/.env.$model -v \
        $PWD/out:/app/out:rw \
        -e MODEL_NM=$model \
        rtm_pose_deploy:latest
done
