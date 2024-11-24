from os import getenv
from torch import save
import transformers
from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM
from accelerate.utils import BnbQuantizationConfig
import torch
from typing import Optional, Union, Dict
import os
from pathlib import Path

def load_quantized_model(
    model_name_or_path: str,
    quantization_bits: int = 4,
    device_map: str = "auto",
    custom_quantization_config: Optional[Dict] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    **kwargs
) -> AutoModelForCausalLM:
    """
    Load a transformer model with specified quantization settings.
    
    Args:
        model_name_or_path (str): HuggingFace model name or path to local model
        quantization_bits (int): Number of bits for quantization (2, 4, or 8)
        device_map (str): Device mapping strategy ("auto", "cpu", "cuda", etc.)
        custom_quantization_config (Dict, optional): Custom quantization configuration
        cache_dir (str, optional): Directory to cache downloaded models
        trust_remote_code (bool): Whether to trust remote code when loading models
        **kwargs: Additional arguments passed to from_pretrained
    
    Returns:
        AutoModelForCausalLM: Loaded and quantized model
        
    Raises:
        ValueError: If quantization_bits is not supported
        RuntimeError: If CUDA is not available when required
        Exception: For other loading errors
    """
    try:
        # Validate quantization bits
        if quantization_bits not in [2, 4, 8]:
            raise ValueError(f"Quantization bits must be 2, 4, or 8. Got {quantization_bits}")
            
        # Check CUDA availability if using GPU
        if device_map != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but GPU device map was specified")
            
        # Set up quantization configuration
        if custom_quantization_config:
            bnb_config = BnbQuantizationConfig(**custom_quantization_config)
        else:
            # Default configuration based on bit width
            load_in_4bit = quantization_bits == 4
            load_in_8bit = quantization_bits == 8
            
            if quantization_bits == 2:
                # Special handling for 2-bit quantization
                bnb_config = BnbQuantizationConfig(
                    load_in_2bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True
                )
            else:
                kwargs.update({
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit
                })
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            quantization_config=bnb_config if custom_quantization_config else None,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        return model
        
    except Exception as e:
        raise Exception(f"Error loading quantized model: {str(e)}")

def quantize_and_save_model(
    model_name: str,
    output_dir: str,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    device_map: str = "auto",
    use_double_quant: bool = True,
    bits: int = 4,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    **kwargs
) -> None:
    """
    Load a model, quantize it, and save both the model and its quantization config.
    
    Args:
        model_name (str): HuggingFace model name or path to local model
        output_dir (str): Directory to save the quantized model
        compute_dtype (torch.dtype): Computation dtype (default: torch.bfloat16)
        quant_type (str): Quantization type ("nf4" or "fp4")
        device_map (str): Device mapping strategy
        use_double_quant (bool): Whether to use double quantization
        bits (int): Number of bits for quantization (default: 4)
        cache_dir (str, optional): Directory to cache downloaded models
        trust_remote_code (bool): Whether to trust remote code when loading models
        **kwargs: Additional arguments passed to from_pretrained
    
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If saving fails
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        if bits not in [2, 4, 8]:
            raise ValueError(f"Bits must be 2, 4, or 8. Got {bits}")
        
        if quant_type not in ["nf4", "fp4"]:
            raise ValueError(f"Quant type must be 'nf4' or 'fp4'. Got {quant_type}")
        
        # Configure quantization
        quantization_config = BnbQuantizationConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            load_in_2bit=(bits == 2),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type
        )
        
        print(f"Loading model {model_name} with {bits}-bit quantization...")
        
        # Load and quantize the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        print(f"Saving quantized model to {output_dir}...")
        
        # Save the model and configuration
        model.save_pretrained(output_dir)
        quantization_config.save_pretrained(output_dir)
        
        # Save a README with the quantization details
        readme_content = f"""
# Quantized Model Information

Original model: {model_name}
Quantization configuration:
- Bits: {bits}
- Compute dtype: {compute_dtype}
- Quantization type: {quant_type}
- Double quantization: {use_double_quant}
- Device map: {device_map}

Generated on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
        """
        
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme_content)
            
        print("Successfully saved quantized model and configuration!")
        
        # Print memory usage if using CUDA
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    except Exception as e:
        raise RuntimeError(f"Failed to quantize and save model: {str(e)}")

if __name__ == "__main__":
    mod=transformers.AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct",token="hf_token=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",load_in_4bit=True)
    # mod=transformers.AutoModelForCausalLM.from_pretrained("crumb/nano-mistral")
    env=getenv("OUTPUT_MODEL")
    quantize_and_save_model(mod, f"{env}/model.pt", quant_type="nf4", bits=4)
    # save(mod, f"{env}/mod.pt")

    # mod=transformers.AutoModelForCausalLM.from_pretrained("crumb/nano-mistral")
    env=getenv("OUTPUT_MODEL")
    # save(mod, f"{env}/mod.pt")

