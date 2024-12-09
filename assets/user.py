import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.utils import simple_parse_args_string
import torch.nn.functional as F

@register_model("user")
class UserLM(LM):
    def __init__(self, args) -> None:
        super().__init__()
        args_dict = simple_parse_args_string(args)
        pretrained_path = args_dict.get('pretrained', 'szymonrucinski/Llama-3.1-8B-AWQ-4bit')
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls(arg_string)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        
        with torch.no_grad():
            for instance in tqdm(requests):
                context, continuation = instance.args
                
                context_tokens = self.tokenizer(context, return_tensors="pt").to(self.model.device)
                full_tokens = self.tokenizer(context + continuation, return_tensors="pt").to(self.model.device)
                
                target_len = len(self.tokenizer(continuation, add_special_tokens=False)['input_ids'])
                target_ids = full_tokens.input_ids[0, -target_len:]
                
                outputs = self.model(**full_tokens)
                logits = outputs.logits
                
                shift_logits = logits[0, -(target_len+1):-1, :]
                shift_labels = target_ids
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs[torch.arange(len(shift_labels)), shift_labels]
                
                sequence_log_prob = token_log_probs.sum().item()
                
                is_greedy = True
                pred_tokens = torch.argmax(shift_logits, dim=-1)
                if not torch.all(pred_tokens == shift_labels):
                    is_greedy = False
                
                results.append((sequence_log_prob, is_greedy))
                
                del outputs, logits, log_probs
                torch.cuda.empty_cache()
        
        return results

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError()