#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer
from sparseml.transformers import SparseAutoModelForCausalLM, apply
from datasets import load_dataset
import pandas as pd

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

def load_calibration_dataset():
    """
    Load the calibration dataset from HuggingFace parquet file.
    Returns:
        dataset: HuggingFace dataset object containing the calibration data
    """
    # Load dataset directly using datasets library
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "https://huggingface.co/datasets/szymonrucinski/calibration-dataset/resolve/main/calibration_dataset/train-00000-of-00001.parquet"
        }
    )
    
    # Select only the text column
    dataset = dataset.select_columns(['text'])
    
    return dataset

def quantize_model(args):
    """
    Quantize a model using SparseML with the specified parameters.
    Args:
        args: Parsed command line arguments
    """
    print(f"Loading model from {args.input_model}...")
    model = SparseAutoModelForCausalLM.from_pretrained(
        args.input_model,
        torch_dtype=torch.bfloat16
    )

    # Load calibration dataset
    print("Loading calibration dataset...")
    dataset = load_calibration_dataset()
    splits = {
        "calibration": "train[:100%]",  # Use full training set for calibration
        "train": "test"                 # Required by SparseML but not used
    }

    print(f"Applying quantization using recipe: {args.recipe}")
    apply(
        model=model,
        dataset=dataset,
        recipe=args.recipe,
        output_dir=args.output_model,
        splits=splits,
        max_seq_length=args.max_length,
        num_calibration_samples=args.num_samples,
    )
    
    print(f"Model quantized and saved to {args.output_model}")

def main():
    args = parse_args()
    quantize_model(args)

if __name__ == "__main__":
    main()