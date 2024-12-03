import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def get_document_embeddings(filepath: str|Path) -> np.ndarray:
    arr = np.load(filepath) # NOTE may want to use memory mapping
    assert len(arr.shape) == 2
    return arr

def create_document_embeddings(model: SentenceTransformer, documents: list[list[str]], path_to_dir: str, name: str) -> np.ndarray:
    
    assert model.get_sentence_embedding_dimension() is not None

    embeddings = np.ndarray((len(documents), model.get_sentence_embedding_dimension()))

    for i, doc in enumerate(documents):
        embeddings[i, :] = model.encode(doc, precision='float32', normalize_embeddings=True).mean(axis=0)

    np.save(f"{path_to_dir}/{name}.npy", embeddings)
    return embeddings


