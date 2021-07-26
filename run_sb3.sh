#!/usr/bin/env bash

IMAGE=mjlbach/ray-sb3-repro
[ -d results/stable-baselines3 ] || mkdir -p results/stable-baselines3

docker build -t $IMAGE . \
    && echo BUILD SUCCESSFUL

# podman build -t $IMAGE . \
#     && echo BUILD SUCCESSFUL

# For docker
docker run --gpus all -ti --rm \
-v $(pwd)/ig_dataset:/opt/iGibson/igibson/data/ig_dataset \
-v $(pwd)/igibson.key:/opt/iGibson/igibson/data/igibson.key \
-v $(pwd)/results:/results \
$IMAGE:latest python trainers/stable-baselines3_example.py

# Or if your cluster uses podman
# podman run --rm -it --net=host \
# --security-opt=no-new-privileges \
# --security-opt label=type:nvidia_container_t \
# -e DISPLAY \
# -v $(pwd)/ig_dataset:/opt/iGibson/igibson/data/ig_dataset \
# -v $(pwd)/igibson.key:/opt/iGibson/igibson/data/igibson.key \
# -v $(pwd)/results/stable-baselines3:/opt/iGibson/igibson/log_dir \
# $IMAGE:latest python trainers/stable-baselines3_example.py
