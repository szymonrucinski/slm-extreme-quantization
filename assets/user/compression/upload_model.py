#!/usr/bin/env python3
import argparse
from huggingface_hub import HfApi, create_repo
import os
from tqdm import tqdm

def upload_model(model_dir: str, repo_name: str, token: str, private: bool = False):
    """
    Upload a model to HuggingFace Hub from a local directory.
    Args:
        model_dir (str): Path to the model directory
        repo_name (str): Name of the HuggingFace repository
        token (str): HuggingFace API token
        private (bool): Whether to create a private repository
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Directory does not exist: {model_dir}")

    # Initialize Hugging Face API
    api = HfApi(token=token)

    try:
        # Try to create repository (will skip if it already exists)
        print(f"Creating/checking repository: {repo_name} ({'private' if private else 'public'})")
        create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True
        )

        # Upload all files in the directory
        print("Starting upload...")
        files = []
        for root, _, filenames in os.walk(model_dir):
            for filename in filenames:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, model_dir)
                files.append((local_path, relative_path))

        # Upload files with progress bar
        for local_path, relative_path in tqdm(files, desc="Uploading files"):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=relative_path,
                repo_id=repo_name,
                token=token
            )

        print(f"\nModel successfully uploaded to: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the model directory to upload"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Name of the HuggingFace repository (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=True,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default is public)"
    )

    args = parser.parse_args()
    upload_model(args.model_dir, args.repo_name, args.hf_token, args.private)

if __name__ == "__main__":
    main()