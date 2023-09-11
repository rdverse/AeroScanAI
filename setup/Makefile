SHELL := /bin/bash # Use bash syntax

install-python:
	sudo apt install software-properties-common
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt install python3.9
	sudo apt install python3-pip

install-python-packages:
	pip install pandas numpy

install-dockertools:
	sudo apt-get update
	sudo apt install docker.io
	sudo apt  install docker-compose
	sudo gpasswd -a $(USER) docker
	newgrp docker

setup-box-defect_classify:
	# create model folder
	mkdir -p ../box/models/defect_classify
	# create data folder
	mkdir -p ../box/datasets/defect_classify
	# create output folder 
	mkdir -p ../box/outputs/defect_classify

setup-box-waveform_probe:
	# create model folder
	mkdir -p ../box/models/waveform_probe
	# create data folder
	mkdir -p ../box/datasets/waveform_probe
	# create output folder 
	mkdir -p ../box/outputs/waveform_probe

setup-box-scan_anomaly:
	# create model folder
	mkdir -p ../box/models/scan_anomaly
	# create data folder
	mkdir -p ../box/datasets/scan_anomaly
	# create output folder 
	mkdir -p ../box/outputs/scan_anomaly

# GENERAL INSTALL
install-tools: install-python install-python-packages install-dockertools


#######DEFECT CLASSIFY#################
create-synthethic-data-dc:
	python3 ../defect_classify/src/utils/generate_data.py --save_path '../box/datasets/defect_classify/'

launch-defect_classify: setup-box-defect_classify create-synthethic-data-dc
	docker-compose -f docker-compose-defect_classify.yaml -p dc up

launch-defect_classify_no_data: 
	docker-compose -f docker-compose-defect_classify.yaml -p dc up

shutdown-defect_classify:
	docker-compose -f docker-compose-defect_classify.yaml -p dc down

###########WAVEFORM PROBE#################
create-synthethic-data-wp:
	python3 ../waveform_probe/src/utils/generate_data.py --save_path '../box/datasets/waveform_probe/'

launch-waveform_probe: setup-box-waveform_probe create-synthethic-data-wp
	docker-compose -f docker-compose-waveform_probe.yaml -p wp up

launch-waveform_probe_no_data: 
	docker-compose -f docker-compose-waveform_probe.yaml -p wp up

shutdown-waveform_probe:
	docker-compose -f docker-compose-waveform_probe.yaml -p wp down


############## SCAN ANOMALY#####################
create-synthethic-data-sa:
	python3 ../scan_anomaly/src/utils/generate_data.py --save_path '../box/datasets/scan_anomaly/'

launch-scan_anomaly: setup-box-scan_anomaly #create-synthethic-data-sa
	docker-compose -f docker-compose-scan_anomaly.yaml -p sa up

launch-scan_anomaly_no_data:
	docker-compose -f docker-compose-scan_anomaly.yaml -p sa up

shutdown-scan_anomaly:
	docker-compose -f docker-compose-scan_anomaly.yaml -p sa down
	
############ DEFECT CLASSIFY AND WAVEFORM PROBE #################

launch-defect_classify_and_waveform_probe: setup-box-defect_classify setup-box-waveform_probe create-synthethic-data-dc create-synthethic-data-wp
	docker-compose -f docker-compose-dc-wp.yaml -p dc-wp up -d

launch-defect_classify_and_waveform_probe_no_data: 
	docker-compose -f docker-compose-dc-wp.yaml -p dc-wp up -d

shutdown-defect_classify_and_waveform_probe:
	docker-compose -f docker-compose-dc-wp.yaml -p dc-wp down


############ DEFECT CLASSIFY AND WAVEFORM PROBE AND SCAN ANOMALY#################
launch-defect_classify_and_waveform_probe_and_scan_anomaly: setup-box-defect_classify setup-box-waveform_probe setup-box-scan_anomaly create-synthethic-data-dc create-synthethic-data-wp create-synthethic-data-sa
	docker-compose -f docker-compose-dc-wp-sa.yaml -p dc up -d

launch-defect_classify_and_waveform_probe_and_scan_anomaly_no_data: 
	docker-compose -f docker-compose-dc-wp-sa.yaml -p dc up -d

shutdown-defect_classify_and_waveform_probe_and_scan_anomaly:
	docker-compose -f docker-compose-dc-wp-sa.yaml -p dc down

############ DEFECT CLASSIFY AND WAVEFORM PROBE AND SCAN ANOMALY#################
launch-all: setup-box-defect_classify setup-box-waveform_probe setup-box-scan_anomaly
	docker-compose -f docker-compose-all.yaml -p all up -d

launch-defect_classify_and_waveform_probe_and_scan_anomaly_no_data: 
	docker-compose -f docker-compose-all.yaml -p all up -d

shutdown-defect_classify_and_waveform_probe_and_scan_anomaly:
	docker-compose -f docker-compose-all.yaml -p all down


#######APP UI#################
launch-app_ui: 
	docker-compose -f docker-compose-app_ui.yaml -p aui up

shutdown-app_ui:
	docker-compose -f docker-compose-app_ui.yaml -p aui down
