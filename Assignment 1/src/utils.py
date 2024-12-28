from preprocessing import *

def term_frequency(word: str, document_frequency: dict) -> float:
    return document_frequency[word] / max(document_frequency.values())

def tf_idf(word: str, document_vocabulary: dict, idf: dict) -> float:
    return term_frequency(word, document_vocabulary)*idf[word]

def avg_doc_length(corpus_frequencies):
    total_length = 0
    count = 0
    for doc_id, doc_freq in corpus_frequencies.items():
        total_length += sum(list(doc_freq.values()))
        count += 1
    return total_length/count

def bm25(query,idf,doc_freq,avg_length,k1,b):
    doc_length = sum(list(doc_freq.values()))
    bm_score = 0
    for word in query.keys():
        if word in idf:
            if word in doc_freq:
                bm_score += idf[word]*(doc_freq[word]*(k1 + 1))/(doc_freq[word] + k1*(1-b + b*doc_length/avg_length))
    return bm_score