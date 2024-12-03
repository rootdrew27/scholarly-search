from pathlib import Path
import sys

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer

from embedding_library import EmbeddingLibrary
# Set paths

PATH_TO_PAPERS_EMB = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/embeddings')

PATH_TO_PAPERS_TXT = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/paper_texts')

PATH_TO_LOG = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/log')
PATH_TO_LOG.mkdir(parents=True, exist_ok=True)

PATH_TO_SEMSIM_WEIGHTS = r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/weights/semsim'

end_of_paper_words = [
    'references',
    'acknowledgements', 
    'acknowledgement',
    'author information',
    'acknowledgment',
    'future work',
    'appendix',
    'funding',
    'conflict of interest',
    'copyright',
    'data availability',
    'acknowledgments',
    'ethical approval',
    
]

papers_to_skip = [
    '2311.11329v2_content.txt',
    '2411.09324v2_content.txt',
    '2411.14259v1_content.txt',
    '2309.01837v3_content.txt'
]
# Intialize models
LLM_Tokenizer = PreTrainedTokenizerFast.from_pretrained("weights/hugging_pt/tokenizer")
LLM = AutoModelForCausalLM.from_pretrained("weights/hugging_pt/model").to('cuda')

SemSim = SentenceTransformer(PATH_TO_SEMSIM_WEIGHTS, device='cpu')

# Init Embedding Library 
embLib = EmbeddingLibrary(
    path_to_papers=PATH_TO_PAPERS_TXT,
    path_to_embs=PATH_TO_PAPERS_EMB,
    model=SemSim,
    end_of_paper_words=end_of_paper_words,
    papers_to_skip=papers_to_skip,
    norm_embs=True,
    path_to_log=PATH_TO_LOG,
    name="Prototype"
)   

embLib.set_paper_embs()

def paperLink(paper_id):
    arxiv_link = f"https://arxiv.org/abs/{paper_id}"
    pdf_link = f"https://arxiv.org/abs/{paper_id}"

    return arxiv_link, pdf_link
def llmPrompt(prompt):

    prompt= f"Give a scholarly response to the question \"{prompt}\". Make the response an IEEE paper. Make it so each section starts with \"SECTION: \" on each part of the paper. Do not include references. Only use academic papers. Inculde nothing additional.\n"


    inputs = LLM_Tokenizer(prompt, return_tensors="pt").to('cuda')

    ids = LLM.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            pad_token_id=LLM_Tokenizer.eos_token_id,
            max_length=4000,
            temperature=0.01
    )

    output = LLM_Tokenizer.batch_decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    
    return output[0]

if __name__ == '__main__':
    filepath = sys.argv[1]
    
    with open(f"./Questions/{filepath}", "r") as file:
        prompts = file.readlines()
    responses = []
    
    
    for prompt in prompts:
        
        responses.append(llmPrompt(prompt))

    print('Switching LLM to RAM and SemSim model to VRAM\n')
    LLM.to('cpu')
    SemSim.to('cuda')
    
    for i, response in enumerate(responses, start=0):
        print("question" + prompts[i])
        print(response)
        prompts[i] = f"Give a scholarly response to the question \"{prompts[i]}\". Make the response an IEEE paper. Make it so each section starts with \"SECTION: \" on each part of the paper. Do not include references. Only use academic papers. Inculde nothing additional.\n"
        top_paper_ids = embLib.search_papers(prompts[i], response, n_results=5)

        paper_links = [paperLink(id)[0] for id in top_paper_ids]
   
        print('Top Results') 
        for i, pl in enumerate(paper_links):
            print(f'{i}: {pl}') 
         
 
