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

# GENERAL INSTALL
install-tools: install-python install-python-packages install-dockertools

# FOR DEFECT CLASSIFY
create-synthethic-data:
	python3 ../defect_classify/src/utils/generate_data.py --save_path '../box/datasets/defect_classify/'

#######DEFECT CLASSIFY#################
launch-defect_classify: setup-box-defect_classify create-synthethic-data 
	docker-compose -f docker-compose-defect_classify.yaml -p dc up

launch-defect_classify_no_data: 
	docker-compose -f docker-compose-defect_classify.yaml -p dc up

shutdown-defect_classify:
	docker-compose -f docker-compose-defect_classify.yaml -p dc down

###########WAVEFORM PROBE#################
launch-waveform_probe: 
	docker-compose -f docker-compose-waveform_probe.yaml -p wp up

launch-waveform_probe_no_data: 
	docker-compose -f docker-compose-waveform_probe.yaml -p wp up

shutdown-waveform_probe:
	docker-compose -f docker-compose-waveform_probe.yaml -p wp down

	
############ DEFECT CLASSIFY AND WAVEFORM PROBE #################

#######APP UI#################
launch-app_ui: 
	docker-compose -f docker-compose-app_ui.yaml -p aui up

shutdown-app_ui:
	docker-compose -f docker-compose-app_ui.yaml -p aui down