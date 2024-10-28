from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch

# BM25 function to get top N ranked documents
def get_top_n_documents(query, corpus, top_n=5):
    """
    Retrieve the top N ranked documents from the corpus for a given query using BM25.
    
    Args:
        query (str): The query text.
        corpus (list of str): List of documents.
        top_n (int): The number of top documents to retrieve.

    Returns:
        list of tuples: Each tuple contains (document, score).
    """
    # Tokenize the corpus
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    
    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize the query
    tokenized_query = query.lower().split()
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Rank documents by score and select top N
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_documents = [(corpus[i], scores[i]) for i in ranked_indices]
    
    return top_documents

# Example usage of get_top_n_documents
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A fast brown fox leaps over a sleeping dog",
    "Brown foxes are quick and clever animals"
]
query = "quick brown fox"
top_n = 2

top_documents = get_top_n_documents(query, corpus, top_n)
print("Top N ranked documents by BM25:")
for i, (doc, score) in enumerate(top_documents, 1):
    print(f"{i}. (Score: {score:.2f}) - {doc}")

