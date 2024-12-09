import torch

from torch.utils.data import DataLoader

from pathlib import Path

import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

from sentence_transformers.trainer import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

from sentence_transformers.training_args import BatchSamplers

from datasets import Dataset

import sys

import os

import logging

from embedding_library import EmbeddingLibrary

from custom_losses import MultipleNegativesRankingLoss

path_to_papers = Path(r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/data/paper_texts')

path_to_embs = Path(r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/data/embeddings')

path_to_log = Path(r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/log')

# should all be lower case

end_of_paper_words = [

    'references',

    'acknowledgement', 

    'appendix',

]



papers_to_skip = [

    '2311.11329v2_content.txt',

    '2411.09324v2_content.txt',

    '2411.14259v1_content.txt',

    '2309.01837v3_content.txt'

    #'2403.12778v2_content.txt' 

]

model_name = 'all-distilroberta-v1'

device = 'cuda'

model = SentenceTransformer(model_name, device=device)

emb_lib = EmbeddingLibrary(

    path_to_papers=path_to_papers,

    path_to_embs=path_to_embs,

    model=model,

    end_of_paper_words=end_of_paper_words,

    papers_to_skip=papers_to_skip,

    norm_embs=True,

    name=model_name,

    log_lvl=logging.INFO,

    path_to_log=path_to_log

)

emb_lib.update_paper_list()

# create paper chunks and Dataset



paper_chunks = []



for paper_path in emb_lib.paper_paths:

    with open(paper_path, 'r', encoding='utf-8') as p:

        paper = p.read()

        title, chunks = emb_lib.preprocess_paper(paper)

        paper_chunks.extend(chunks)

        paper_chunks.append(title)



dataset = Dataset.from_dict({

    "anchor": paper_chunks,

    "positive": paper_chunks,

})

batch_size = 1000

all_texts = np.random.shuffle([dataset[i : i + batch_size]["anchor"] for i in range(0, len(dataset), batch_size)])



def batch_iterator():

    for i in range(0, len(dataset), batch_size):

        yield dataset[i : i + batch_size]["anchor"]



model.tokenizer = model.tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=50265)

from torch.nn.modules.dropout import Dropout



def set_dropout(model, p):



    def set_d(module):

        for c in module.children():

            if len(list(module.children())) == 0: return

            if isinstance(c, Dropout):

                c.p = p

            set_d(c)



    for m in model.modules():

        set_d(m)



set_dropout(model, 0.5)   

train_loss = MultipleNegativesRankingLoss(model, kl_factor=0.5)

args = SentenceTransformerTrainingArguments(

    output_dir=f"training/{model_name}",

    num_train_epochs=10,

    per_device_train_batch_size=64,

    #per_device_eval_batch_size=16,

    warmup_ratio=0.1,

    fp16=True,  # Set to False if your GPU can't handle FP16

    bf16=False,  # Set to True if your GPU supports BF16

    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates

    # Optional tracking/debugging parameters:

    # eval_strategy="steps",

    # eval_steps=100,

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,

    logging_steps=100,

    run_name=f"{model_name}",  # Used in W&B if `wandb` is installed

)

trainer = SentenceTransformerTrainer(

    model = model,

    args = args,

    train_dataset = dataset,

    loss = train_loss,

    tokenizer = model.tokenizer,  

)

trainer.train()

path_to_weights = r'/data/users/roota5351/my_cs426/group_5/SchSearch/scholarly-search/weights/semsim3'

model.save(path_to_weights)
