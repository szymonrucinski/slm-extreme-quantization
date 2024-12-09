#!/bin/bash
set -x  # Print commands for debugging
set -e  # Exit on error

# Environment variables expected:
# HF_TOKEN: Hugging Face access token
# OUTPUT_MODEL: Directory for final model files

# Login to Hugging Face
# huggingface-cli login --token ${HF_TOKEN}
huggingface-cli login --token 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
# Install dependencies
# python3 setup.py install
pip3 install "sparseml[transformers]"
# Update Transformers to solve
# ValueError: `rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
pip3 install transformers==4.46.3
pip3 install autoawq
pip3 install lm-eval
pip3 install --upgrade aqlm[gpu,cpu]


if [[ "$1" == "--compress" ]]; then
    echo "Generate calibration dataset.
    Calculate perplexity score for each element between <100;512> tokens.
    Get samples across the entire perplexity range. 10 buckets with 100 per bucket.
    "
    # python3 0-make_calibration_set.py
    
    echo "Running Initial Eval of LLama-3.1-8B model"
    HF_TOKEN="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"
    OUTPUT_MODEL="meta-llama/Llama-3.2-1B"
    # ./compression/custom_eval.sh "$HF_TOKEN" "$ORIGINAL_MODEL" "$OUTPUT_MODEL"

    echo "Running 4-bit quantization of the model using AWQ W4A16 with custom calibration dataset"
    python3 ./compression/1-awq-4-bit.py --input-model $OUTPUT_MODEL --output-model ./compression/output_model/Llama-3.1-8B-AWQ-4bit
    python3 upload_model.py --model-dir ./compression/output_model/Llama-3.1-8B-AWQ-4bit --repo-name "szymonrucinski/Llama-3.1-8B-AWQ-4bit" 
    OUTPUT_MODEL="meta-llama/Llama-3.1-8B"
    ./compression/custom_eval.sh "$HF_TOKEN" "$ORIGINAL_MODEL" "./compression/output_model/Llama-3.1-8B-AWQ-4bit"


    echo "Running pruning of 4-bit quantization of the model using AWQ W4A16 "
    python3 upload_model.py --model-dir ./compression/output_model/Llama-3.1-8B-AWQ-4bit --repo-name "szymonrucinski/Llama-3.1-8B-AWQ-4bit" 

    
    # Verify output directory exists
    if [[ ! -d "$OUTPUT_MODEL" ]]; then
        echo "Error: OUTPUT_MODEL directory not found after compression"
        exit 1
    fi
fi

echo "BUILD SUCCESSFUL"