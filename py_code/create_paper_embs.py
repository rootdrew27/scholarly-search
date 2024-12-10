from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer, SimilarityFunction

from sentence_transformers.models.Normalize import Normalize

import logging

import traceback

import re

from embedding_library import EmbeddingLibrary


PATH_TO_PAPERS_EMB = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/embeddings')

PATH_TO_PAPERS_TXT = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/data/paper_texts')

PATH_TO_LOG = Path(r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/log')

PATH_TO_LOG.mkdir(parents=True, exist_ok=True)

PATH_TO_SEMSIM_WEIGHTS = r'/data/classes/2024/fall/cs426/group_5/SchSearch/scholarly-search/weights/semsim4'

end_of_paper_words = [
    'references',
    'acknowledgement',
    'author information',
    'acknowledgment',
    'future work',
    'appendix',
    'funding',
    'conflict of interest',
    'copyright',
    'data availability',
    'acknowledgment',
    'ethical approval', 
]

papers_to_skip = [
    '2311.11329v2_content.txt',
    '2411.09324v2_content.txt',
    '2411.14259v1_content.txt',
    '2309.01837v3_content.txt'
]

SemSim = SentenceTransformer(PATH_TO_SEMSIM_WEIGHTS, device='cuda')

embLib = EmbeddingLibrary(
    path_to_papers=PATH_TO_PAPERS_TXT,
    path_to_embs=PATH_TO_PAPERS_EMB,
    model=SemSim,
    end_of_paper_words=end_of_paper_words,
    papers_to_skip=papers_to_skip,
    norm_embs=True,
    path_to_log=PATH_TO_LOG,
    name="BLAH2"
)  

embLib.embed_papers(skip_existing=False)
