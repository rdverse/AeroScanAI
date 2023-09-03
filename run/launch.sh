#!/bin/bash

bash docker_kill.sh

cd ../setup

# shutdown all running modules
make shutdown-app_ui

# select what module to launch
make launch-app_ui
#make launch-defect_classify 
#make launch-defect_classify_no_data


