import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np


def run_data_filtering():
    MODEL_ID = "meta-llama/Llama-3.1-8B"
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 512
    CSV_FILE = './calibration_data/dclm_perplexities_512.csv'
    
    print("Loading DCLM-micro dataset...")
    ds = load_dataset("robbiegwaldd/dclm-micro")
    
    print(f"Loading {MODEL_ID} for perplexity scoring...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    
    results = []
    texts = ds['train']['text']
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Filter by token length
        valid_texts, valid_indices = filter_by_token_length(
            batch_texts, 
            tokenizer, 
            min_length=100, 
            max_length=SEQUENCE_LENGTH
        )
        
        if not valid_texts:
            continue
        
        batch_perplexities = calculate_batch_perplexity(valid_texts, model, tokenizer)
        
        # Save results using original indices
        for local_idx, perp in zip(valid_indices, batch_perplexities):
            results.append({
                'index': i + local_idx,
                'text': batch_texts[local_idx][:1000],  # Truncate text for CSV
                'perplexity': float(perp),
                'token_length': len(tokenizer(batch_texts[local_idx])['input_ids'])
            })
        
        # Save progress periodically
        if len(results) % 100 == 0:
            df = pd.DataFrame(results)
            df.to_csv(CSV_FILE, index=False)
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILE, index=False)
    
    print(f"\nResults saved to {CSV_FILE}")
    print("\nDataset statistics:")
    print(f"Total sequences processed: {len(df)}")
    print(f"Average perplexity: {df['perplexity'].mean():.2f}")
    print(f"Median perplexity: {df['perplexity'].median():.2f}")
    print(f"Average token length: {df['token_length'].mean():.2f}")
    print("\nTop 5 sequences by perplexity:")
    print(df.nsmallest(5, 'perplexity')[['index', 'token_length', 'perplexity']])
    
    del model
    torch.cuda.empty_cache()
    


###
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
import os

def filter_by_token_length(texts, tokenizer, min_length=100, max_length=512):
    """Filter texts by their tokenized length."""
    valid_texts = []
    valid_indices = []
    
    for idx, text in enumerate(texts):
        tokens = tokenizer(text, truncation=False)['input_ids']
        token_length = len(tokens)
        if min_length <= token_length <= max_length:
            valid_texts.append(text)
            valid_indices.append(idx)
    
    return valid_texts, valid_indices

def calculate_batch_perplexity(texts, model, tokenizer, max_length=512):
    """Calculate perplexity for a batch of texts."""
    try:
        encodings = tokenizer(
            texts,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**encodings)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = encodings['input_ids'][..., 1:].contiguous()
            shift_mask = encodings['attention_mask'][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
            
            losses = losses.view(shift_labels.size()) * shift_mask
            seq_lengths = shift_mask.sum(dim=1)
            seq_losses = losses.sum(dim=1) / seq_lengths
            perplexities = torch.exp(seq_losses).cpu().numpy()
            
        return perplexities
    except Exception as e:
        print(f"Error in batch: {e}")
        return np.array([float('inf')] * len(texts))

def create_perplexity_buckets(samples, perplexity_scores, num_buckets=10, samples_per_bucket=100):
    # Remove top 20% highest perplexity samples
    threshold = np.percentile(perplexity_scores, 80)
    mask = perplexity_scores < threshold
    filtered_samples = np.array(samples)[mask]
    filtered_scores = perplexity_scores[mask]
    
    # Create bucket boundaries
    min_perp = np.min(filtered_scores)
    max_perp = np.max(filtered_scores)
    bucket_boundaries = np.linspace(min_perp, max_perp, num_buckets + 1)
    
    # Distribute samples into buckets
    selected_samples = []
    selected_indices = []
    bucket_stats = []
    
    for i in range(num_buckets):
        lower = bucket_boundaries[i]
        upper = bucket_boundaries[i+1]
        
        # Find samples in this perplexity range
        bucket_mask = (filtered_scores >= lower) & (filtered_scores < upper)
        bucket_samples = filtered_samples[bucket_mask]
        bucket_scores = filtered_scores[bucket_mask]
        
        # Randomly sample from this bucket
        if len(bucket_samples) > samples_per_bucket:
            indices = np.random.choice(len(bucket_samples), samples_per_bucket, replace=False)
            selected = bucket_samples[indices]
            selected_perp = bucket_scores[indices]
        else:
            selected = bucket_samples
            selected_perp = bucket_scores
            
        selected_samples.extend(selected)
        selected_indices.extend(np.where(bucket_mask)[0])
        
        # Save bucket statistics
        bucket_stats.append({
            'bucket': i,
            'range': f"{lower:.2f}-{upper:.2f}",
            'total_samples': len(bucket_samples),
            'selected_samples': len(selected),
            'avg_perplexity': np.mean(selected_perp)
        })
    
    return selected_samples, selected_indices, bucket_stats

def run_full_pipeline():
    # Constants
    CSV_FILE = 'dclm_perplexities_512.csv'
    OUTPUT_DIR = './calibration_data'
    HF_REPO_ID = "szymonrucinski/calibration-dataset"
    NUM_BUCKETS = 10
    SAMPLES_PER_BUCKET = 100
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load CSV with perplexity scores
    print("Loading perplexity scores from CSV...")
    df = pd.read_csv(f'{OUTPUT_DIR}/{CSV_FILE}')
    
    # Create perplexity buckets
    print("Creating perplexity buckets...")
    selected_samples, selected_indices, bucket_stats = create_perplexity_buckets(
        df['text'].tolist(),
        df['perplexity'].values,
        num_buckets=NUM_BUCKETS,
        samples_per_bucket=SAMPLES_PER_BUCKET
    )
    
    # Save bucket statistics
    stats_df = pd.DataFrame(bucket_stats)
    stats_df.to_csv(f"{OUTPUT_DIR}/bucket_statistics.csv", index=False)
    print("\nBucket Statistics:")
    print(stats_df)
    
    # Create filtered dataset with proper length checking
    print("\nCreating filtered dataset...")
    print(f"Number of selected samples: {len(selected_samples)}")
    print(f"Number of selected indices: {len(selected_indices)}")
    
    # Get perplexities for selected indices
    selected_perplexities = []
    valid_samples = []
    valid_indices = []
    
    # Only keep samples where we have all required data
    for sample, idx in zip(selected_samples, selected_indices):
        if idx < len(df):
            selected_perplexities.append(df['perplexity'].iloc[idx])
            valid_samples.append(sample)
            valid_indices.append(idx)
    
    # Create DataFrame with validated data
    filtered_df = pd.DataFrame({
        'text': valid_samples,
        'perplexity': selected_perplexities,
        'original_index': valid_indices
    })
    
    print(f"Final number of valid samples: {len(filtered_df)}")
    filtered_df.to_csv(f"{OUTPUT_DIR}/filtered_samples.csv", index=False)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(filtered_df)
    
    # Save as PyTorch tensors
    print("Saving as PyTorch dataset...")
    torch.save({
        'texts': valid_samples,
        'perplexities': torch.tensor(selected_perplexities),
        'indices': torch.tensor(valid_indices)
    }, f"{OUTPUT_DIR}/calibration_dataset.pth")
    
    # Push to HuggingFace
    print(f"\nPushing to HuggingFace Hub: {HF_REPO_ID}")
    api = HfApi()
    
    # Push dataset
    dataset.push_to_hub(HF_REPO_ID, "calibration_dataset")
    
    # Push additional files
    api.upload_file(
        path_or_fileobj=f"{OUTPUT_DIR}/bucket_statistics.csv",
        path_in_repo="bucket_statistics.csv",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    
    api.upload_file(
        path_or_fileobj=f"{OUTPUT_DIR}/calibration_dataset.pth",
        path_in_repo="calibration_dataset.pth",
        repo_id=HF_REPO_ID,
        repo_type="dataset"
    )
    
    print("\nPipeline completed successfully!")
    print(f"Total valid samples: {len(filtered_df)}")
    print(f"Files pushed to: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    # Run perplexity calculation
    # run_data_filtering()
    
    # Run full pipeline
    run_full_pipeline()