#!/usr/bin/env python3
from os import getenv
from torch import save
import transformers
from transformers import AutoTokenizer
from sparseml.transformers import SparseAutoModelForCausalLM, apply
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize Llama model using SparseML')
    parser.add_argument(
        '--input-model',
        type=str,
        required=True,
        help='Input model path or Hugging Face model ID'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='Output directory for the quantized model'
    )
    parser.add_argument(
        '--recipe',
        type=str,
        required=True,
        help='Path to the SparseML recipe file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of calibration samples'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model from {args.input_model}...")
    model = SparseAutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch.bfloat16
    )
    
    dataset = "szymonrucinski/calibration-dataset"
    splits = {"calibration": "train[:100%]", "train": "test"}
    
    print(f"Applying quantization using recipe: {args.recipe}")
    apply(
        model=model,
        dataset=dataset,
        recipe=args.recipe,
        bf16=True,
        output_dir=args.output_model,
        splits=splits,
        max_seq_length=args.max_length,
        num_calibration_samples=args.num_samples,
    )
    
    print(f"Model quantized and saved to {args.output_model}")

if __name__ == "__main__":
    main()