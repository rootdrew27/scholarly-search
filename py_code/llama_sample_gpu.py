from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

toker = PreTrainedTokenizerFast.from_pretrained("weights/hugging_pt/tokenizer")
model = AutoModelForCausalLM.from_pretrained("weights/hugging_pt/model").to('cuda')

model.to('cuda')

prompt = r"I'm just a chill"

inputs = toker(prompt, return_tensors="pt").to('cuda')

ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs['attention_mask'],
    pad_token_id=toker.eos_token_id,
    max_length=20,
    temperature=0.1,
    
)

output = toker.batch_decode(
    ids,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)

print(output[0])

