#!/bin/bash

bash docker_kill.sh

cd ../setup

make shutdown-app_ui

make launch-app_ui
