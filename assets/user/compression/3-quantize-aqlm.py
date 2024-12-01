#!/usr/bin/env python3

import argparse
from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aqlm import AQLMQuantizer
from datasets import load_dataset

def set_seeds(seed_value=42):
    """Set seeds for reproducibility"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize Llama model using AQLM')
    parser.add_argument('output_dir', type=str, help='Output directory for the quantized model')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B",
                      help='Model ID to quantize')
    parser.add_argument('--num_codebooks', type=int, default=1,
                      help='Number of codebooks for AQLM')
    parser.add_argument('--codebook_bits', type=int, default=16,
                      help='Codebook bits for AQLM')
    parser.add_argument('--group_size', type=int, default=64,
                      help='Group size for AQLM')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of calibration samples to randomly select')
    return parser.parse_args()

def get_calibration_data(num_samples, seed):
    """Load calibration data from the dataset with random sampling"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Get total dataset size
    total_samples = len(dataset)
    
    # Randomly select indices
    rng = np.random.RandomState(seed)
    selected_indices = rng.choice(total_samples, size=num_samples, replace=False)
    selected_indices = sorted(selected_indices)  # Sort for consistency
    
    # Select the random samples
    calibration_texts = dataset.select(selected_indices)["text"]
    
    print(f"Randomly selected {num_samples} samples from {total_samples} total samples")
    print(f"Using indices from {selected_indices[0]} to {selected_indices[-1]}")
    
    return [text for text in calibration_texts if text.strip()]

def main():
    args = parse_args()
    
    print(f"Setting random seeds to {args.seed} for reproducibility...")
    set_seeds(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("Loading calibration data...")
    calibration_data = get_calibration_data(args.num_samples, args.seed)

    print("Initializing AQLM quantizer...")
    quantizer = AQLMQuantizer(model)

    quant_config = {
        "num_codebooks": args.num_codebooks,
        "codebook_bits": args.codebook_bits,
        "group_size": args.group_size,
        "seed": args.seed
    }

    print("Starting quantization...")
    quantized_model = quantizer.quantize(
        quant_config=quant_config,
        calibration_data=calibration_data,
        tokenizer=tokenizer
    )

    print(f"Saving quantized model to {output_dir}...")
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration including seed and sampling info
    config_file = output_dir / "quantization_config.txt"
    with open(config_file, "w") as f:
        f.write(f"Model ID: {args.model_id}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Number of calibration samples: {args.num_samples}\n")
        f.write(f"Number of codebooks: {args.num_codebooks}\n")
        f.write(f"Codebook bits: {args.codebook_bits}\n")
        f.write(f"Group size: {args.group_size}\n")
        f.write(f"Effective bits per parameter: {args.num_codebooks * args.codebook_bits / args.group_size:.2f}\n")
    
    print("Quantization complete!")
    print(f"Model saved at: {output_dir}")
    print(f"Quantization config saved at: {config_file}")

if __name__ == "__main__":
    main()