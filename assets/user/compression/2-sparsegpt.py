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
    parser.add_argument('output_dir', type=str, help='Output directory for the quantized model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    model = SparseAutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B", 
        torch_dtype=torch.bfloat16
    )

    dataset = "robbiegwaldd/dclm-micro"
    recipe = "/home/admin/test/slm-extreme-quantization/assets/user/compression/2-sparsegpt.py"
    splits = {"calibration": "train[:100%]", "train": "test"}
    max_seq_length = 256
    num_calibration_samples = 512

    apply(
        model=model,
        dataset=dataset,
        recipe=recipe,
        bf16=True,
        output_dir=args.output_dir,
        splits=splits,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

if __name__ == "__main__":
    main()