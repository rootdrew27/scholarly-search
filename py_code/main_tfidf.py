from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from embedding_library import EmbeddingLibrary
import logging

SEMSIM_NAME = 'semsim3'

prompt_format = "Scholarly. Detailed. {}"
#prompt_format = f"Give a scholarly response to the question, \"{}\". Make the response an IEEE paper. Each Section starts with \"SECTION: \". Do not include references. Only use academic papers. Include nothing additional.\n"

# set paths

PATH_TO_PAPER_EMBS = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/embeddings/' + f'{SEMSIM_NAME}')
PATH_TO_PAPERS = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/paper_texts')
PATH_TO_LOG = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/log')
PATH_TO_LOG.mkdir(parents=True, exist_ok=True)
PATH_TO_SEMSIM_WEIGHTS = r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/weights/' + f'{SEMSIM_NAME}'

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

log_lvl_name = sys.argv[2]
log_lvl = logging._nameToLevel[log_lvl_name]
logger = logging.getLogger('Main')
logger.setLevel(log_lvl)
logging.basicConfig(
    filename=PATH_TO_LOG / f'Main_{log_lvl_name}.log',
    level=log_lvl,
    encoding='utf-8',
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Intialize models

LLM_Tokenizer = PreTrainedTokenizerFast.from_pretrained("weights/hugging_pt/tokenizer")
LLM = AutoModelForCausalLM.from_pretrained("weights/hugging_pt/model").to('cuda')
SemSim = SentenceTransformer(PATH_TO_SEMSIM_WEIGHTS, device='cpu')

# Init Embedding Library 

embLib = EmbeddingLibrary(
    path_to_papers=PATH_TO_PAPERS,
    path_to_embs=PATH_TO_PAPER_EMBS,
    model=SemSim,
    end_of_paper_words=end_of_paper_words,
    papers_to_skip=papers_to_skip,
    norm_embs=True,
    logger=logger,    
)   

embLib.update_paper_list()


def paperLink(paper_id):
    arxiv_link = f"https://arxiv.org/abs/{paper_id}"
    pdf_link = f"https://arxiv.org/abs/{paper_id}"

    return arxiv_link, pdf_link

def llmPrompt(prompt):

    prompt = prompt_format.format(prompt)
    inputs = LLM_Tokenizer(prompt, return_tensors="pt").to('cuda')

    output = LLM.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            pad_token_id=LLM_Tokenizer.eos_token_id,
            penalty_alpha=0.6,
            top_k=4,
            max_new_tokens=2048,
    )

    response = LLM_Tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return response

if __name__ == '__main__':
    filename = sys.argv[1] 
    logger.info(f'Starting run for: {filename}')

    with open(f"./questions/{filename}", "r") as file:
        prompts = file.readlines()

    responses = []        
    for prompt in prompts: 
        response = llmPrompt(prompt)
        logger.info(f'Prompt: {prompt}\nResponse: {response}')
        responses.append(response)

    logger.info('Switching LLM to RAM and SemSim model to VRAM\n')
    LLM.to('cpu')
    SemSim.to('cuda')
 
    N_RESULTS = 5 
    for i, response in enumerate(responses):
        print("Input: " + prompts[i])
        top_paper_ids, top_scores = embLib.search_papers(prompt_format.format(prompts[i], response, n_results=N_RESULTS, threshold=0.05)
        if len(top_paper_ids) == 0:
            print(f'There are no good search results. Try another search')

        elif len(top_paper_ids) < N_RESULTS:
            print(f'There are only {len(top_paper_ids)} for your search. Perhaps try another search for more results.')

        paper_links = [paperLink(id)[0] for id in top_paper_ids]

        print('Top Results') 
        for i, pl in enumerate(paper_links):
            print(f'{i}: {pl} - Score = {top_scores[i]}') 
