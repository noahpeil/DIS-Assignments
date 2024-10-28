import numpy as np

def find_top_k_doc(queries: dict,documents: dict,k: int,language: str):
    """Returns the top k documents for each query

    Args:
        queries: {query_id: query_embedding}
        documents: {document_id: document_embedding}
        k: a scalar denoting the number of document to retrieve for each query
        language: the selected language
    
    """
    query_matrix = list()
    queries_ids = list()
    for query_id, query_embeddings in queries.items():
        query_matrix.append(query_embeddings)
        queries_ids.append(query_id)
    query_matrix = np.array(query_matrix)
    queries_norms = np.diagonal(query_matrix.dot(query_matrix.T))
    queries_inverse_norms = np.linalg.inv(np.diag(np.sqrt(queries_norms)))
    
    doc_matrix = list()
    doc_ids = list()
    for doc_id, doc_embeddings in documents.items():
        doc_matrix.append(doc_embeddings)
        doc_ids.append(doc_id)

    if language != "en": #Batch the english documents because of the high number
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
                cosine_similarities[:,i*step:] = sub_cosine_similarities

    top_k_per_query = cosine_similarities.argsort(axis=1)[::-1][:,:k]
    doc_ids = np.array(doc_ids)
    top_k_documents_id = dict()
    for i in range(len(queries_ids)):
        top_k_documents_id[queries_ids[i]] = doc_ids[top_k_per_query[i]].tolist()
    return top_k_documents_id