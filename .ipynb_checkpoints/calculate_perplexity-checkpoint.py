#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np

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

def main():
    MODEL_ID = "meta-llama/Llama-3.1-8B"
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 512
    CSV_FILE = 'dclm_perplexities_512.csv'
    
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

if __name__ == "__main__":
    main()