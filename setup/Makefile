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

install-tools: install-python install-python-packages install-dockertools

 
launch-app_ui: 
	docker-compose -f docker-compose-app_ui.yaml -p app_ui up

launch-app_ui-nodata: 
	docker-compose -f docker-compose-app_ui.yaml -p app_ui up

shutdown-app_ui:
	docker-compose -f docker-compose-app_ui.yaml -p app_ui down
