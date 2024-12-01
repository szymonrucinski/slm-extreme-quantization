import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_model(output_path: str) -> None:
    """
    Quantize a Llama model and save it to the specified output path.
    
    Args:
        output_path (str): Path where the quantized model will be saved
    """
    model_path = "meta-llama/Llama-3.1-8B"
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    # Load model
    print("Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Quantize
    print("Quantizing model...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    # Save quantized model
    print("Saving quantized model...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    print(f'Model is quantized and saved at "{output_path}"')

def main():
    parser = argparse.ArgumentParser(description='Quantize Llama model using AWQ')
    parser.add_argument('output_path', type=str, help='Path where the quantized model will be saved')
    args = parser.parse_args()
    
    quantize_model(args.output_path)

if __name__ == "__main__":
    main()