Monitoring example
Prerequisites

You need following tools installed:

    docker
    docker-compose (included to Docker Desktop for Mac and Docker Desktop for Windows )

Preparation

Note: all actions expected to be executed in repo folder.

    Create virtual environment and activate it (eg. python -m venv venv && source ./venv/bin/activate)
    Install required packages pip install -r requirements.txt


Starting services

To start all required services, execute:

docker-compose up

It will start following services:

    prometheus - TSDB for metrics
    grafana - Visual tool for metrics
    mongo - MongoDB, for storing raw data, predictions, targets and profile reports
    evidently_service - Evindently RT-monitoring service (draft example)
    prediction_service - main service, which makes predictions

Sending data

To start sending data to service, execute:

python send_data.py

This script will send every second single row from dataset to prediction service along with creating file target.csv with actual results (so it can be loaded after)
