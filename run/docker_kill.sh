#!/bin/bash

# Stop and remove all Docker containers
docker stop $(docker ps -a -q)  # Stop all containers
docker rm $(docker ps -a -q)    # Remove all containers

# Delete all Docker images
docker rmi $(docker images -a -q)  # Delete all images

# Print a message indicating completion
echo "All Docker containers stopped and removed. All Docker images deleted."
