from os import getenv
from torch import save
import transformers


if __name__ == "__main__":
    #mod=transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    mod=transformers.AutoModelForCausalLM.from_pretrained("crumb/nano-mistral")
    env=getenv("OUTPUT_MODEL")
    save(mod, f"{env}/mod.pt")

