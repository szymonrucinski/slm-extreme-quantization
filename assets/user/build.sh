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
pip3 install --upgrade aqlm[gpu,cpu]


if [[ "$1" == "--compress" ]]; then
    echo "Generate calibration dataset.
    Calculate perplexity score for each element between <100;512> tokens.
    Get samples across the entire perplexity range. 10 buckets with 100 per bucket.
    "
    python3 0-make_calibration_set.py
	echo "Running SpareGPT pruning 1:4 and quantization using 4-BIT-AWQ quantization"
    # python3 /opt/llm/user/compression/compress.py
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

	#AWQ-4bit
    # python3 /opt/llm/user/compression/1-awq-4-bit.py /opt/llm/user/compression/output_model/Llama-3.1-8B-AWQ-4bit
	# python3 /opt/llm/user/compression/2-sparsegpt.py /opt/llm/user/compression/output_model/Llama-3.1-8B-SparseGPT-4bit
    python3 /opt/llm/user/compression/3-quantize-aqlm.py /opt/llm/user/compression/output_model/Llama-3.1-8B-AQLM-2bit

    
    # Verify output directory exists
    if [[ ! -d "$OUTPUT_MODEL" ]]; then
        echo "Error: OUTPUT_MODEL directory not found after compression"
        exit 1
    fi
fi

echo "BUILD SUCCESSFUL"