from os import getenv
from torch import save
import transformers
from transformers import AutoTokenizer
from sparseml.transformers import SparseAutoModelForCausalLM, oneshot
import torch


if __name__ == "__main__":
    model=transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    recipe = """
    sparsity_stage:
        run_type: oneshot
    sparsity_modifiers:
    SparseGPTModifier:
        sparsity: 0.25
        mask_structure: "1:4"
        sequential_update: false
    quantization_stage:
    run_type: oneshot
    quantization_modifiers:
        GPTQModifier:
        sequential_update: false
        ignore: ["lm_head"]
        config_groups:
            group_0:
            weights:
                num_bits: 4
                type: "int"
                symmetric: true
                strategy: "channel"
                targets: ["Linear"]"""
    
    # model_stub = "zoo:llama2-7b-ultrachat200k_llama2_pretrain-base"
    model = SparseAutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")

    # uses SparseML's built-in preprocessing for ultra chat
    dataset = "wikitext"
    dataset_config_name="wikitext-2-raw-v1",


    # save location of quantized model
    output_dir = "/opt/llm/user/compression/output_model"
    
    # set dataset config parameters
    splits = {"calibration": "train_gen[:10%]"}
    max_seq_length = 768
    pad_to_max_length = False
    num_calibration_samples = 512
    
    oneshot(
    model=model,
    dataset=dataset,
    dataset_config_name=dataset_config_name,
    recipe=recipe,
    output_dir=output_dir,
    splits=splits,
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True
    )