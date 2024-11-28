#!/bin/bash
set -x  # Print commands for debugging
set -e  # Exit on error

# Environment variables expected:
# HF_TOKEN: Hugging Face access token
# OUTPUT_MODEL: Directory for final model files

# Login to Hugging Face
huggingface-cli login --token ${HF_TOKEN}

# Install dependencies
python3 setup.py install
# Install python sparseml


if [[ "$1" == "--compress" ]]; then
    echo "Starting model compression pipeline..."

	echo "Running SpareGPT pruning 1:4"
	sparseml.transformers.text_generation.oneshot \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --recipe "./recipe.yaml" \
    --output_dir "./compression/output_model" \
    --precision float16 \
    --dataset open_platypus \
    --text_column text

	echo "Running model Quantitzaion in 2bit"
	echo "RUN AWQ quantization"

    
    
    # Run additional compression steps
    echo "Running custom compression pipeline..."
    cd compression
    python3 compress.py
    
    # Verify output directory exists
    if [[ ! -d "$OUTPUT_MODEL" ]]; then
        echo "Error: OUTPUT_MODEL directory not found after compression"
        exit 1
    fi
fi

echo "BUILD SUCCESSFUL"