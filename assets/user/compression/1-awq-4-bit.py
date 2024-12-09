import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import pandas as pd
import requests
import os

def download_parquet(url: str, save_path: str) -> str:
    """
    Download parquet file from URL.
    Args:
        url (str): URL to download from
        save_path (str): Path to save the file
    Returns:
        str: Path to downloaded file
    """
    print(f"Downloading calibration dataset from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

def quantize_model(input_path: str, output_path: str) -> None:
    """
    Quantize a model and save it to the specified output path.
    Args:
        input_path (str): Path to the input model
        output_path (str): Path where the quantized model will be saved
    """
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Download and load calibration dataset
    print("Loading calibration dataset...")
    try:
        url = "https://huggingface.co/datasets/szymonrucinski/calibration-dataset/resolve/main/calibration_dataset/train-00000-of-00001.parquet"
        parquet_path = "calibration_dataset/train-00000-of-00001.parquet"
        
        # Download the file if it doesn't exist
        if not os.path.exists(parquet_path):
            download_parquet(url, parquet_path)
        
        # Load the parquet file
        df = pd.read_parquet(parquet_path)
        calib_data = df['text'].tolist()
        print(f"Successfully loaded {len(calib_data)} calibration examples")
        
        if len(calib_data) == 0:
            raise ValueError("No calibration data found")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load calibration dataset: {str(e)}")

    # Load model
    print(f"Loading model from {input_path}...")
    model = AutoAWQForCausalLM.from_pretrained(
        input_path,
        low_cpu_mem_usage=True,
        use_cache=False
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        input_path,
        trust_remote_code=True
    )

    # Quantize
    print("Quantizing model...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    # Save quantized model
    print("Saving quantized model...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    print(f'Model is quantized and saved at "{output_path}"')

def main():
    parser = argparse.ArgumentParser(description='Quantize model using AWQ')
    parser.add_argument(
        '--input-model',
        type=str,
        required=True,
        help='Path to the input model'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        required=True,
        help='Path where the quantized model will be saved'
    )

    args = parser.parse_args()
    quantize_model(args.input_model, args.output_model)

if __name__ == "__main__":
    main()