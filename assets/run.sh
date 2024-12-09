#!/bin/bash -xeu
set -o pipefail

# huggingface-cli login --token ${HF_TOKEN}
lm-eval \
    --model user \
    --model_args pretrained=/workspace/slm-extreme-quantization/assets/user/compression/output_model/Llama-3.1-8B-AWQ-4bit \
    --tasks boolq \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code  | tee output
# or use output_path https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
acc=$(bash mean.sh output)
sz=$(du -s ${OUTPUT_MODEL} | awk '{print $1}') 
score=$(echo "(${acc} / ${sz}) * (1024*1024)" | bc -l)
printf '%3f\n' ${score}
