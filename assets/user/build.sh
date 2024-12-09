#!/bin/bash
set -x # Print commands for debugging
set -e # Exit on error

# Base directory structure
BASE_DIR="/workspace/slm-extreme-quantization/assets/user"
COMPRESSION_DIR="${BASE_DIR}/compression"
OUTPUT_DIR="${COMPRESSION_DIR}/output_model"
HF_TOKEN=${HF_TOKEN:-"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"}


# Environment variables and defaults
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

INPUT_MODEL=${INPUT_MODEL:-"meta-llama/Llama-3.1-8B"}
MODEL_REPO="szymonrucinski/Llama-3.1-8B-AWQ-4bit-huawei"
OUTPUT_MODEL_NAME="Llama-3.1-8B-AWQ-4bit"
FINAL_OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_MODEL_NAME}"

# Function to check command success
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Function to ensure directories exist
ensure_directories() {
    mkdir -p "${OUTPUT_DIR}"
    check_command "Directory creation"
}

# Function to download the model
download_model() {
    echo "Downloading model from Hugging Face..."
    huggingface-cli download --token "${HF_TOKEN}" \
        "${MODEL_REPO}" \
        --local-dir "${OUTPUT_DIR}" \
        --local-dir-use-symlinks False
    check_command "Model download"
}

# Function to install dependencies
install_dependencies() {
    pip3 install "sparseml[transformers]"
    pip3 install transformers==4.46.3
    pip3 install autoawq
    pip3 install lm-eval
    pip3 install --upgrade "aqlm[gpu,cpu]"
    check_command "Dependencies installation"
}

# Function to login to Hugging Face
hf_login() {
    huggingface-cli login --token "${HF_TOKEN}"
    check_command "Hugging Face login"
}

# Function to run compression
run_compression() {
    # echo "Running Initial Eval of ${INPUT_MODEL}"
    # python3 "${COMPRESSION_DIR}/custom_eval.sh" "${HF_TOKEN}" "${INPUT_MODEL}" "${FINAL_OUTPUT_DIR}"
    check_command "Initial evaluation"
    
    echo "Running 4-bit quantization using AWQ W4A16 with custom calibration dataset"
    python3 "${COMPRESSION_DIR}/1-awq-4-bit.py" \
        --input-model "${INPUT_MODEL}" \
        --output-model "${FINAL_OUTPUT_DIR}"
    check_command "AWQ quantization"
    
    python3 "${COMPRESSION_DIR}/upload_model.py" \
        --model-dir "${FINAL_OUTPUT_DIR}" \
        --repo-name "${MODEL_REPO}" \
        --hf-token "${HF_TOKEN}"
    check_command "Model upload"
    
    # python3 "${COMPRESSION_DIR}/custom_eval.sh" \
    #     "${HF_TOKEN}" \
    #     "${INPUT_MODEL}" \
    #     "${FINAL_OUTPUT_DIR}"
    # check_command "Final evaluation"
}

# Main execution
main() {
    ensure_directories
    install_dependencies
    hf_login
    download_model
    
    if [[ "$1" == "--compress" ]]; then
        run_compression
        
        if [[ ! -d "${FINAL_OUTPUT_DIR}" ]]; then
            echo "Error: Output directory not found after compression"
            exit 1
        fi
    fi
    
    echo "BUILD SUCCESSFUL"
}

# Execute main with all arguments
main "$@"