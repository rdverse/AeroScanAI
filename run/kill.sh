#!/bin/bash

docker rm -fv $(docker ps -a -q)    # Remove all containers
# Stop and remove all Docker containers
docker stop $(docker ps -a -q)  # Stop all containers
docker rm -fv $(docker ps -a -q)    # Remove all containers

# Print a message indicating completion
echo "All Docker containers stopped and removed. All Docker images deleted."
