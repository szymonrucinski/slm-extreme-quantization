set -e
set -u
docker build --no-cache . --build-arg hf_token=$(cat ~/.cache/huggingface/token) -t huawei-llm:0.1
