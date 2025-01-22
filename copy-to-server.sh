#!/bin/bash

# Define the server details
SERVER="ec-gregorjt@fox.educloud.no"

# Copy the files
scp sanity.py sample.slurm train.py ${SERVER}:~/