import numpy as np
import torch

def find_top_k_doc(queries_list: list, queries_embeddings: torch.Tensor, documents_list: list, documents_embeddings: torch.Tensor, k: int, language: str):
    """Returns the top k documents for each query

    Args:
        queries: {query_id: query_embedding}
        documents: {document_id: document_embedding}
        k: a scalar denoting the number of document to retrieve for each query
        language: the selected language
    
    """

    """if language != "en": #Batch the english documents because of the high number
        doc_matrix = np.array(doc_matrix)
        documents_norms = np.diagonal(doc_matrix.dot(doc_matrix.T))
        documents_inverse_norms = np.linalg.inv(np.diag(np.sqrt(documents_norms)))
        cosine_similarities = np.dot(queries_inverse_norms,np.dot(np.dot(query_matrix,doc_matrix.T),documents_inverse_norms))
        
    else:
        step = len(doc_matrix)//10
        cosine_similarities = np.zeros((len(queries_ids),len(doc_ids)))
        for i in range(10):
            if i != 9:
                doc_sub_matrix = np.array(doc_matrix[i*step:(i+1)*step])
            else:
                doc_sub_matrix = np.array(doc_matrix[i*step:])
            documents_norms = np.diagonal(doc_sub_matrix.dot(doc_sub_matrix.T))
            documents_inverse_norms = np.linalg.inv(np.diag(np.sqrt(documents_norms)))
            sub_cosine_similarities = np.dot(queries_inverse_norms,np.dot(np.dot(query_matrix,doc_sub_matrix.T),documents_inverse_norms))
            if i != 9:
                cosine_similarities[:,i*step:(i+1)*step] = sub_cosine_similarities
            else:
                cosine_similarities[:,i*step:] = sub_cosine_similarities"""
    
    queries_inverse_norms = torch.rsqrt(torch.sum(queries_embeddings * queries_embeddings, dim=1, keepdim=True))
    documents_inverse_norms = torch.rsqrt(torch.sum(documents_embeddings * documents_embeddings, dim=1, keepdim=True))

    cosine_similarities = queries_embeddings.dot(documents_embeddings.T)
    cosine_similarities *= queries_inverse_norms
    cosine_similarities *= documents_inverse_norms.T

    top_k_per_query = cosine_similarities.argsort(dim=1, descending=True)[:,:k]
    documents_list = np.array(documents_list)
    top_k_documents_id = dict()
    for i in range(len(queries_list)):
        top_k_documents_id[queries_list[i]] = documents_list[top_k_per_query[i]].tolist()
    return top_k_documents_id