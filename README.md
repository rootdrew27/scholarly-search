# Scholarly Search
---

The aim of this repository is to develop a system for improving search in research papers, through semantics.
This project does not rely strongly on keywords and intends to utilize DL model's awareness of context and semantics.


The process is as follows:
1. The user enters a prompt pertaining to their research interest/question.
2. The prompt is injected into a system prompt format and passed to an LLM (finetuned on the research papers).
3. The output of the model is embedded by a Semantic Similarity model (also finetuned on the research papers).
4. The similarity between the resulting embedding and the embeddings of the papers (calculated prior) is calculated.
5. The title and URL of the most similar papers are returned to the user.


**Terms used to fetch papers**
- Autoencoder
- Llama
- GPT
- GAN
- Semantic Similarity
- Machine Learning
- Deep Learning
- Transformer
- Vision Transformer
- Supervised Learning
- Unsupervised Learning
- Random Forest
- AI