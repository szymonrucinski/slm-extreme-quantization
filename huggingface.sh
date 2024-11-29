#!/bin/bash
# Login to Hugging Face
echo "Logging into Hugging Face..."
huggingface-cli login
mkdir -p results
# Variables to track best model
best_sparsity=""
best_accuracy=0
best_dir=""
# First create all recipe files
for sparsity in 0.25 0.50 0.75; do
    recipe_file="recipe_sparsity_${sparsity}.yaml"
    # Create recipe file with proper YAML indentation
    cat > "${recipe_file}" << EOF
oneshot_stage:
  pruning_modifiers:
    SparseGPTModifier:
      sparsity: ${sparsity}
      sequential_update: true
      mask_structure: 0:0
      targets: ['re:model.layers.\d*$']
EOF
done
# Create header for final results file
echo "Sparsity | Accuracy" > results/all_results.txt
echo "----------|----------" >> results/all_results.txt
# Then run the evaluation loop
for sparsity in 0.25 0.50 0.75; do
    recipe_file="recipe_sparsity_${sparsity}.yaml"
    output_dir="pruned_${sparsity}"
    echo "Running evaluation for sparsity ${sparsity}"
    # Run pruning with proper model configuration
    sparseml.transformers.text_generation.oneshot \
        --model meta-llama/Llama-3.1-8B \
        --recipe "${recipe_file}" \
        --dataset wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --output_dir "./${output_dir}" \
        --precision float16 \
        --dataset wikitext \
        --text_column text
    # Ensure model type is set in config
    if [ -f "./${output_dir}/config.json" ]; then
        # Add model_type if it doesn't exist
        python3 -c "
import json
with open('./${output_dir}/config.json', 'r') as f:
    config = json.load(f)
if 'model_type' not in config:
    config['model_type'] = 'llama'
with open('./${output_dir}/config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
    fi
    # Run evaluation
    lm_eval \
        --model hf \
        --model_args pretrained="./${output_dir}" \
        --tasks wikitext \
        --device cuda:0 \
        --batch_size 8 \
        --seed 7 > "results/post_pruning_${sparsity}.txt"
    # Extract accuracy and append to results file
    accuracy=$(grep "acc" "results/post_pruning_${sparsity}.txt" | awk '{print $7}')
    if [ ! -z "$accuracy" ]; then
        echo "${sparsity} | ${accuracy}" >> results/all_results.txt
        # Check if this is the best model so far
        if (( $(echo "$accuracy > $best_accuracy" | bc -l) )); then
            best_accuracy=$accuracy
            best_sparsity=$sparsity
            # If there was a previous best directory, remove it
            if [ ! -z "$best_dir" ]; then
                rm -rf "$best_dir"
            fi
            # Save the new best directory with a consistent name
            best_dir="best_model"
            rm -rf "$best_dir"
            cp -r "$output_dir" "$best_dir"
        fi
    else
        echo "${sparsity} | N/A" >> results/all_results.txt
    fi
    # Remove output directory after evaluation
    rm -rf "./${output_dir}"
    echo "Cleaned up ${output_dir}"
done
# Find and append best result
echo -e "\nBest Result:" >> results/all_results.txt
sort -t'|' -k2 -nr results/all_results.txt | head -n 1 >> results/all_results.txt
# Display final results
cat results/all_results.txt
# Push best model to Hugging Face if one was found
if [ ! -z "$best_dir" ] && [ -d "$best_dir" ]; then
    echo "Pushing best model (sparsity: ${best_sparsity}, accuracy: ${best_accuracy}) to Hugging Face..."
    # Create model card
    cat > "${best_dir}/README.md" << EOF
# Pruned Llama 3.1 8B
This is a pruned version of Llama 3.1 8B with ${best_sparsity} sparsity, achieving ${best_accuracy} accuracy on evaluation tasks.
## Model Details
- Base model: meta-llama/Llama-3.1-8B
- Pruning method: SparseGPT
- Sparsity: ${best_sparsity}
- Evaluation accuracy: ${best_accuracy}
- Tasks evaluated: boolq, wikitext, winogrande, mmlu
## Usage
This model can be used as a standard Hugging Face transformer model.
## Training Details
- Pruning technique: One-shot pruning using SparseML
- Dataset used: open_platypus
- Hardware: Single GPU training
EOF
    # Push to Hugging Face
    repo_name="pruned-llama-3.1-8b-${best_sparsity}"
    python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi()
# Create the repository
api.create_repo(
    repo_id='szymonrucinski/${repo_name}',
    private=True,
    exist_ok=True
)
# Upload the model files
api.upload_folder(
    folder_path='${best_dir}',
    repo_id='szymonrucinski/${repo_name}',
    commit_message='Upload pruned model with ${best_sparsity} sparsity'
)
print(f'Model successfully uploaded to szymonrucinski/${repo_name}')
"
    echo "Model successfully pushed to Hugging Face"
    # Clean up best model directory
    rm -rf "$best_dir"
else
    echo "No valid model found to push to Hugging Face"
fi