from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction
from sentence_transformers.models.Normalize import Normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from logging import Logger
import traceback
import copy
import re

class EmbeddingLibrary():
    def __init__(self, path_to_papers:Path, path_to_embs:Path, model:SentenceTransformer, end_of_paper_words:list[str], papers_to_skip:list[str], norm_embs:bool, logger, path_to_log:Path=None, log_lvl=logging.INFO, log_encoding='utf-8'):
        self.path_to_papers:Path = path_to_papers
        self.path_to_embs:Path = path_to_embs
        self.model = model
        if norm_embs: model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
        if isinstance(model._last_module(), Normalize): norm_embs = False # this prevents the model from normalizing twice
        self.emb_size = model.encode(['']).shape[-1]
        self.paper_embs = None
        self.paper_ids:list[str] = sorted([file.stem.replace('_content', '') for file in path_to_papers.iterdir() if file.name not in papers_to_skip])
        self.paper_paths:list[Path] = sorted([file for file in path_to_papers.iterdir() if file.name not in papers_to_skip])
        #self.path_to_clean_papers = None TODO: Implement (perhaps create a PaperLibrary class)
        self.end_of_paper_words:list[str] = end_of_paper_words
        self.end_of_paper_regex = re.compile('(' + "|".join([f'SECTION: (\d+.?)*{word}s?' for word in end_of_paper_words]).lstrip('|') + ').*', flags=re.IGNORECASE|re.DOTALL)
        self.norm_embs = norm_embs

        if isinstance(logger, Logger):
            self.logger = logger
        elif isinstance(logger, str):
            self.logger = logging.getLogger(f'{logger}_embs_{logging._levelToName[log_lvl]}')
            self.path_to_log = path_to_log
            self.logger.setLevel(log_lvl)
            logging.basicConfig(
                filename=self.path_to_log / f'{logger}_embs_{logging._levelToName[log_lvl]}.log',
                level=log_lvl,
                encoding=log_encoding
            )
        else:
            raise Exception(f'Logger must be a logging.Logger or a str but it was a {type(logger)}')

        self.tfidf_vectorizer = TfidfVectorizer()
        self.paper_embs_tfidf = None

    def remove_links(self, chunk):
        chunk = re.sub(r'(\d+)?http[s]?:\/\/[^\s]+([^\.\,:; ])', r'', chunk) # remove [] citations
        # TODO: remove (blah et. al.) citations
        return chunk
    
    def remove_citations(self, chunk):
        chunk = re.sub(r'\[(\d+[,]?)+\]([^\w\s])', r'\2', chunk) # for when punc follow cite
        chunk = re.sub(r'\[(\d+[,]?)+\](\w+)', r' \2', chunk) # for when chars follow cite
        chunk = re.sub(r'\[(\d+[,]?)+\]\s', r'', chunk) # for when nothing follows the cite
        chunk = re.sub(r'\(\w+ et\.? al\.\)(\w+)', r'\1', chunk) # for non-IEEE cites
        return chunk

    def preprocess_paper(self, paper: str):
        title = re.search(r'SECTION:\s+[\d\.]*(.*)\n', paper, flags=re.IGNORECASE).group(1)
        paper = re.sub(self.end_of_paper_regex, r'', paper) # remove ending
        chunks = [
            self.remove_citations(self.remove_links(chunk))
            for sec in re.split(r'SECTION:\s+[\d\.]*.*\n', paper)
            for chunk in sec.strip('\n').split('\n')
            if chunk != ''
        ]
        paper_chunks = [self.remove_citations(self.remove_links(chunk)) for chunk in chunks]

        return title, paper_chunks

    def is_paper_good(self, paper:str, file:Path):
        # check for section in first line
        # check for usage of SECTION
        # check for section names: introduction, conclusion
        try:
            has_title = 'SECTION:' in paper.split('\n', 1)[0]
            re.match(r'SECTION:\s*[\d\.]*\s+\n', paper)
            n_secs = len(re.findall(r'SECTION:', paper))
            paper_lower = paper.lower()
            n_important_secs = len(re.findall(r'section: [\d\.]*[introduction|methodology|methods|conclusions|conclusion|results|experimental results and analysis|experimental results]', paper_lower))
            
            good_paper = True if n_secs > 4 and n_important_secs > 2 and has_title else False
            if not good_paper:
                self.paper_paths.remove(file)
                self.paper_ids.remove(file.stem.replace('_content', ''))
        except Exception as ex:
            print('Exception caught in is_paper_good')
            raise ex

        return good_paper

    def update_paper_list(self):
        paper_paths = copy.deepcopy(self.paper_paths)
        for paper_path in paper_paths:
            with open(paper_path, 'r', encoding='utf-8') as f:
                paper = f.read()
                self.is_paper_good(paper, paper_path)

    def set_paper_embs(self):
        if not isinstance(self.paper_embs, np.ndarray):
            paper_embs = []
            for file in sorted(self.path_to_embs.iterdir()):
                paper_embs.append(np.load(file))
            all_embs = np.concatenate(paper_embs, axis=0)
            assert all_embs.shape[0] == len(self.paper_ids), f'The shape of the full paper embeddings ({all_embs.shape[0]}) does not equal the number of papers ({len(self.paper_ids)}).'
            self.paper_embs = all_embs
        return

    def embed_papers_tfidf(self):

        # collect paper strings and fit_transform
        papers = []
        for paper_path in self.paper_paths:
            with open(paper_path, 'r', encoding='utf-8') as f:
                title, chunks = self.preprocess_paper(f.read())
                p_text = title + " " + " ".join(chunks)
                papers.append(p_text)
        self.paper_embs_tfidf = self.tfidf_vectorizer.fit_transform(papers)

    def search_papers_tfidf(self, prompt:str, n_results=5):
        prompt_emb = self.tfidf_vectorizer.transform([prompt])
        scores = cosine_similarity(prompt_emb, self.paper_embs_tfidf)
        top_idxs = np.argsort(scores[0]).tolist()[-n_results:]
        return [self.paper_ids[idx] for idx in top_idxs]


    def embed_papers(self, skip_existing:bool):
        try:
            if skip_existing:
                existing_embs = [name.stem for name in self.path_to_embs.iterdir()]
                papers_list = []
                for p in self.paper_paths:
                    if p.stem.replace('_content', '') not in existing_embs:
                        papers_list.append(p)
            else:
                papers_list = copy.deepcopy(self.paper_paths)
            
            skip_count = 0 # does not include papers_to_skip
            self.logger.info(f'STARTING EMBEDDING PROCESS!\n')
            self.logger.info(f'The paper_list has {len(papers_list)} papers.')
            self.logger.info(f'The paper_list = {papers_list}')
            for file in papers_list:
                with open(file, 'r', encoding='utf-8') as f:
                    
                    paper = f.read()
                    if not self.is_paper_good(paper, file):
                        self.logger.info(f"SKIPPING {file.name}\n");
                        skip_count+=1
                        continue

                    self.logger.info(f'STARTING TO EMBED: {file.name}')

                    title, chunks = self.preprocess_paper(paper)
                    title_emb = self.model.encode(title, normalize_embeddings=self.norm_embs).reshape(1, self.emb_size)
                    chunk_embs = self.model.encode(chunks, normalize_embeddings=self.norm_embs).reshape(-1, self.emb_size)

                    self.logger.info(f'END OF FILE - SAVING EMBEDDINGS FOR {file.name}\n')
                    paper_emb_mean = np.concatenate([title_emb, chunk_embs]).mean(axis=0)
                    paper_emb = (paper_emb_mean / np.linalg.norm(paper_emb_mean)).reshape(-1, self.emb_size) # normalize paper emb
                    self.logger.info(f'Embedding Shape = {paper_emb.shape}')
                    np.save(self.path_to_embs / file.stem.replace('_content', ''), paper_emb)
 
            self.logger.info(f'{self.paper_ids}')
            self.logger.info('Ending Embedding Process')
            self.logger.info(f'Num of papers skipped = {skip_count} (not including papers_to_skip attribute).')

            assert len(self.paper_ids) == len(list(self.path_to_embs.iterdir())), f"The num of paper_ids {len(self.paper_ids)} must match the num of embs {len(list(self.path_to_embs.iterdir()))}"

            assert self.paper_ids == [file.stem for file in sorted(self.path_to_embs.iterdir())], f'The paper IDs and the embeddings IDs must be in the same order.\nPaper IDs = {self.paper_ids}\nEmbedding IDs = {[file.stem for file in sorted(self.path_to_embs.iterdir())]}'

        except Exception as ex:
            self.logger.error(f'Error with th embedding process: {ex}')
            raise ex

    def preprocess_llm_response(self, prompt:str, response:str) -> list[str]:
        response = response.replace(prompt, '').strip()
        response = re.sub(self.end_of_paper_regex, r'', response)
        response_secs = re.split(r'SECTION:.*\n', response)
        response_chunks = [
            re.sub(r'[\d\*\#]+', r'', self.remove_citations(self.remove_links(chunk)).strip(' '))
            for response_sec in response_secs
            for chunk in re.split(r'\n', response_sec.strip(' \n'))
            if chunk != ''
        ] 
        return response_chunks

    def search_papers(self, prompt:str, response:str, n_results=5, threshold=0.25):
        assert self.paper_embs is not None, print("self.paper_embs must be set to use this function!")
        r_secs = self.preprocess_llm_response(prompt, response)
        self.logger.info(f'The preprocessed LLM response = {r_secs}\n')
        r_emb = self.model.encode(r_secs, normalize_embeddings=self.norm_embs).mean(axis=0).astype(np.float32)
        scores = self.model.similarity(r_emb, self.paper_embs.astype(np.float32))
        scores[scores < threshold] = 0.0
        top_n_idxs:list = np.argsort(scores).tolist()[0][-n_results:]
        top_scores = np.sort(scores[0]).tolist()[-n_results:]
        top_scores.reverse()
        top_n_idxs.reverse()
        top_paper_ids = [self.paper_ids[idx] for idx in top_n_idxs if scores[0, idx] != 0.0]
        return top_paper_ids, top_scores

     
# OLD WORK
# class EmbeddingLibrary():
#     def __init__(self, path_to_papers:Path, path_to_embs:Path, model:SentenceTransformer, end_of_paper_words:list[str], papers_to_skip:list[str], name:str, path_to_log:Path, log_lvl=logging.INFO):
#         self.path_to_papers:Path = path_to_papers
#         self.path_to_embs:Path = path_to_embs
#         self.model = model
#         self.emb_size = model.encode(['']).shape[-1]
#         self.paper_ids:list[str] = [file.stem.replace('_content', '') for file in path_to_papers.iterdir() if file.name not in papers_to_skip]
#         self.path_to_log = path_to_log
#         self.log_lvl:int = log_lvl
#         self.name:str = name
#         self.end_of_paper_words:list[str] = end_of_paper_words # TODO Change to use regex
#         self.full_paper_embs:np.ndarray = None

#     def preprocess(self, content: str):
#         # remove links (do not capture (most) punctuation at the end)
#         content = re.sub(r'(\d+)?http[s]?:\/\/[^\s]+([^\.\,:; ])', r'', content) 
#         # remove [] citations
#         content = re.sub(r'\[(\d+[,]?)+\]([^\w\s])', r'\2', content) # for when punc follow cite
#         content = re.sub(r'\[(\d+[,]?)+\](\w+)', r' \2', content) # for when chars follow cite
#         return content

#     def is_paper_good(self, paper:str):
#         # check for usage of SECTION
#         # check for section names: introduction, conclusion
#         n_secs = len(re.findall(r'SECTION:', paper))
#         paper_lower = paper.lower()
#         n_important_secs = len(re.findall(r'section: [\d\.]*[introduction|methodology|methods|conclusions|conclusion|results|experimental results and analysis|experimental results]', paper_lower))

#         return True if n_secs > 4 and n_important_secs > 2 else False
        

    # def multi_sec_embed(self, skip_existing:bool, norm_embs:bool, encoding='utf-8'):
    
    #     logger = logging.getLogger(f'{self.name}_embs_{logging._levelToName[self.log_lvl]}')
    #     logger.setLevel(self.log_lvl)
    #     logging.basicConfig(
    #         filename=self.path_to_log / f'{self.name}_embs_{logging._levelToName[self.log_lvl]}.log',
    #         level=self.log_lvl,
    #         encoding=encoding
    #     )

    #     try:
    #         if skip_existing:
    #             papers_list = [p for p in self.path_to_papers.iterdir() if p.replace('_content', '') not in self.paper_ids]
    #         else:
    #             papers_list = list(self.path_to_papers.iterdir())

    #         logger.info(f'STARTING EMBEDDING PROCESS!\n')
    #         for file in papers_list:
    #             try:
    #                 with open(file, 'r', encoding='utf-8') as f:

    #                     # read in the content, split into sections, extract and embed the title 
    #                     paper = f.read()
    #                     if not self.is_paper_good(paper): continue

    #                     paper_chunks = paper.split('\n')
    #                     logger.info(f'STARTING TO EMBED: {file.name}')
    #                     title = paper_chunks[0]

    #                     logger.info(f'SKIPPING FILE {file.name}\n');
    #                     if 'introduction' in title.lower(): continue 
    #                     else: title = title.replace('SECTION: ', '')

    #                     title_emb = self.model.encode(title, normalize_embeddings=norm_embs)
    #                     paper_embs = title_emb.reshape(1, self.emb_size)

    #                     n_secs = 1 # start at one for title
    #                     sec = [] # a list of all sentences in a section
    #                     all_secs = []
    #                     for chunk in paper_chunks[1:]:
    #                         if chunk == '':
    #                             continue
                            
    #                         elif 'SECTION' in chunk:

    #                             if any([word in chunk.lower() for word in self.end_of_paper_words]):
    #                                 break

    #                             if len(sec) > 0:
    #                                 logger.debug('EMBEDDING PREVIOUOS SECTION')
    #                                 n_secs += 1
    #                                 sec_embs = self.model.encode(sec, normalize_embeddings=norm_embs)
    #                                 sec_emb = sec_embs.mean(axis=0)
    #                                 all_secs.append(sec_embs)
    #                                 logging.info(f'NUM OF EMBS IN SECTION {n_secs} = {len(sec)}')
    #                                 logging.debug(f'SEC EMB SHAPE: {sec_emb.shape}')
    #                                 paper_embs = np.concatenate([paper_embs, sec_emb.reshape(1, self.emb_size)], axis=0)
    #                                 logging.debug(f'CURRENT PAPER_EMB SHAPE: {paper_embs.shape}')
    #                                 logger.debug(f'STARTING NEW SECTION:')
    #                                 logger.debug(f'TITLE: {chunk}\n')
    #                                 sec = []
    #                             continue

    #                         else:
    #                             # preprocess the chunk
    #                             logger.debug(f'CHUNK BEING PREPROCESSED:\n{chunk}\n')
    #                             chunk = self.preprocess(chunk)
    #                             sec.append(chunk)
    #                             logger.debug(f'PREPROCESSED CHUNK:\n{chunk}\n')

    #                 logger.info(f'END OF FILE - SAVING EMBEDDINGS FOR {file.name}\n')
    #                 full_paper_emb = np.concatenate(all_secs).mean(axis=0).reshape(1, self.emb_size)
    #                 paper_embs = np.concatenate([paper_embs, full_paper_emb], axis=0)
    #                 np.save(self.path_to_embs / file.stem.replace('_content', ''), paper_embs)
    #                 assert paper_embs.shape[0] - 1 == n_secs, f"Num of embeddings {paper_embs.shape[0] - 1} does NOT match num of sections {n_secs}"

    #             except AssertionError as assEx:
    #                 logger.error(assEx)
    #                 logger.error(f'Exception caught while embedding file:{file.stem}')
    #                 continue
    #         return None
    #     except Exception as ex:
    #         print('SHIT')
    #         print(ex)
    #         raise ex
