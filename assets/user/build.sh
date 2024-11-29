#!/bin/bash
set -x  # Print commands for debugging
set -e  # Exit on error

# Environment variables expected:
# HF_TOKEN: Hugging Face access token
# OUTPUT_MODEL: Directory for final model files

# Login to Hugging Face
huggingface-cli login --token ${HF_TOKEN}
# Install dependencies
# python3 setup.py install
pip3 install "sparseml[transformers]"
# Update Transformers to solve
# ValueError: `rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
pip3 install transformers==4.46.3

if [[ "$1" == "--compress" ]]; then
    echo "Starting model compression pipeline..."

	echo "Running SpareGPT pruning 1:4 and quantization using 4-BIT-AWQ quantization"
    python3 /opt/llm/user/compression/compress.py
    echo "Running SpareGPT pruning 1:4 and quantization using 2-BIT-AQLM quantization"


    # sparseml.transformers.text_generation.oneshot \
    # --model meta-llama/Llama-3.1-8B \
    # --recipe "./recipe.yaml" \
    # --dataset_config_name wikitext-2-raw-v1 \
    # --output_dir "./compression/output_model" \
    # --precision float16 \
    # --dataset wikitext \
    # --text_column text
    
    # # Run additional compression steps
    # echo "Running custom compression pipeline..."
    # cd compression
    python3 /opt/user/llm/compression/compress.py
    
    # Verify output directory exists
    if [[ ! -d "$OUTPUT_MODEL" ]]; then
        echo "Error: OUTPUT_MODEL directory not found after compression"
        exit 1
    fi
fi

echo "BUILD SUCCESSFUL"