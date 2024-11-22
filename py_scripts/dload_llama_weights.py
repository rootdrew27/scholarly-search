import transformers
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, LlamaTokenizerFast  

tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.1-70B")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")

tokenizer.save_pretrained('weights/70B/tokenizer', 'tokenizer')

model.save_pretrained('weights/70B/model', 'model')
