#!/bin/bash

#bash docker_kill.sh

cd ../setup

make shutdown-app_ui
make shutdown-defect_classify
make shutdown-defect_classify_and_waveform_probe
make shutdown-defect_classify_and_waveform_probe_and_scan_anomaly

