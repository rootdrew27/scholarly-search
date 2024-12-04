Gomes' Instructions:

- Only one person per group using the notebook.
- Try with CPU first.
- GPU usage is reserved for batch scripts only.

Important Dates:

Nov 19:
You should have:
- Dataset
- Input -> Output
- Models
- Parameters to tune
- Data Imbalance?
- Objective
- 

Nov 26:
- Results

Dec 3:
- Results on Cluster

**Semantic Similarity**

- Use an SS model with a large context window.
- Use a SS model that compares documents to documents (rather than a query-to-document comparsion). 
- Document embeddings could be hierarchical. An embedding for each and an embedding for the entire document.


Test 1
- Prepare 10 unbiased questions and 10 (or more semi-biased questions)
- label 10-50 documents as answering one or more of these questions.
- Determine a cut-off for similarity score. The final model will be expected to beat this score when determining the similarity between an LLM's output and the correct document(s).


Test 2
- Have unbiased users prompt our application with a question (related to computer science research). 
- The user will look at the top 5 results returned and rate each one (0-3) in terms of the relevancy, where 0 is not relevant, 1 is not very relevant, 2 is somewhat relevant, and 3 is relevant.

