#!/bin/bash

# stop and remove all containers listed in models/container_names.txt

# get all container names
for container in $(cat models/container_names.txt); do
  echo "Stopping container $container"
  docker stop $container
  echo "Removing container $container"
  docker rm $container
done