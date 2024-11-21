#!/bin/bash

# Navigate to the model_data directory
cd ~/program_self/year4/AIST2010GROUP/project/models/model_data || exit

# Loop through each unique model version (e.g., v1, v2, ...)
for model in $(ls v*_model_e*.pth | awk -F'_model_e' '{print $1}' | sort | uniq); do
    # Find the latest epoch number for the current model
    latest_epoch=$(ls ${model}_model_e*.pth 2>/dev/null | \
                   awk -F'_model_e' '{print $2}' | \
                   awk -F'.pth' '{print $1}' | \
                   sort -n | tail -1)
    
    # Construct the filename of the latest epoch
    latest_file="${model}_model_e${latest_epoch}.pth"
    
    echo "Keeping: ${latest_file}"
    
    # Find and delete all other epoch files except the latest one
    ls ${model}_model_e*.pth 2>/dev/null | grep -v "_e${latest_epoch}.pth" | \
    while read -r file; do
        echo "Deleting: ${file}"
        rm -v "$file"
    done
done
