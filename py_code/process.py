from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

toker = PreTrainedTokenizerFast.from_pretrained("weights/hugging_pt/tokenizer")
model = AutoModelForCausalLM.from_pretrained("weights/hugging_pt/model").to('cuda')

def llmPrompt(prompt):

    prompt= f"Give a scholarly response to the question \"{prompt}\". Make the response an IEEE paper. Make it so each section starts with \"SECTION: \" on each part of the paper including references. Only use academic papers. Inculde nothing additional"


    inputs = toker(prompt, return_tensors="pt").to('cuda')

    ids = model.generate(inputs.input_ids, max_length=1000)

    output = toker.batch_decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output[0]
print(llmPrompt('how do vision transformers work and how does it affect the up scaling'))