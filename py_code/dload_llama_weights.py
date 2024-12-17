import transformers
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, LlamaTokenizerFast  
from pathlib import Path


path_to_llama_weights = r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/weights/meta-llama/Meta-Llama-3.1-70B/original/consolidated.00.pth'
path_to_tokenizer = r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/weights/meta-llama/Meta-Llama-3.1-70B/original/tokenizer.model'

#if path_to_llama_weights.exists() and path_to_tokenizer.exists():
#    print(f'Llama and it\'s tokenizer are already downloaded\nIf you need to re-download, please delete the files at {path_to_llama_weights.parent()}')
#    print('Exiting Program')
#    exit()

tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

tokenizer.save_pretrained('./weights/llama-3.3-70B/tokenizer', 'tokenizer')

model.save_pretrained('./weights/llama-3.3-70B/model', 'model')
