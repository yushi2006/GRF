#!/bin/bash

# This script runs the full ablation study for the hierarchical fusion model.
# It iterates through different modality orderings and evaluation settings (2-class and 7-class).

CONFIG_FILE="configs/mosi_regression.yaml"

# Define the modality orderings to test
ORDERS=(
    "audio vision text"
    "text vision audio"
    "text audio vision"
)

# Define the number of classes for evaluation
NUM_CLASSES=(2 7)

# Flag to ensure profiling is only done once
PROFILED=false

echo "================================================="
echo "STARTING GRF ABLATION STUDY"
echo "================================================="

for nc in "${NUM_CLASSES[@]}"; do
    echo -e "\n----- RUNNING EXPERIMENTS FOR $nc-CLASS EVALUATION -----"
    for order in "${ORDERS[@]}"; do
        echo -e "\n>>> Training with order: $order <<<"
        
        CMD="python train.py --config $CONFIG_FILE --num_classes $nc --order $order"
        
        # Add the --profile flag only for the very first run
        if [ "$PROFILED" = false ]; then
            CMD="$CMD --profile"
            PROFILED=true
        fi
        
        # Execute the command
        eval $CMD
    done
done

echo "================================================="
echo "ALL EXPERIMENTS FINISHED."
echo "View results with: mlflow ui"
echo "================================================="