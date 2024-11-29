from os import getenv
from torch import save
import transformers
from transformers import AutoTokenizer
from sparseml.transformers import SparseAutoModelForCausalLM, apply
import torch


if __name__ == "__main__":
    model=transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    
    # model_stub = "zoo:llama2-7b-ultrachat200k_llama2_pretrain-base"
    model = SparseAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)

    # uses SparseML's built-in preprocessing for ultra chat
    dataset = "wikitext"
    dataset_config_name="wikitext-2-raw-v1"
    recipe="/opt/llm/user/compression/recipe.yaml"

    # save location of quantized model
    output_dir = "/opt/llm/user/compression/output_model"
    
    # set dataset config parameters
    splits = {"calibration": "train[:100%]", "train": "test"}
    max_seq_length = 1024
    pad_to_max_length = False
    num_calibration_samples = 2048
    
    apply(
        model=model,
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        recipe=recipe,
        bf16=True,
        output_dir=output_dir,
        splits=splits,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        # num_train_epochs=num_train_epochs,
        # logging_steps=logging_steps,
        # save_steps=save_steps,
        # gradient_checkpointing=gradient_checkpointing,
        # learning_rate=learning_rate,
        # lr_scheduler_type=lr_scheduler_type,
        # warmup_ratio=warmup_ratio,
    )