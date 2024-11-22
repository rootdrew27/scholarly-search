from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

toker = PreTrainedTokenizerFast.from_pretrained("weights/hugging_pt/tokenizer")
model = AutoModelForCausalLM.from_pretrained("weights/hugging_pt/model")

#model.to('cuda') # for running on GPU

prompt = "Chill guy?"

inputs = toker(prompt, return_tensors="pt")#.to('cuda')

ids = model.generate(inputs.input_ids, max_length=10) # TODO: Pass in attn mask

output = toker.batch_decode(
    ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output[0])
