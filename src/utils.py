import collections
import numpy as np
from preprocessing import *

def get_document_vocabulary(words: str) -> dict:
    return dict(collections.Counter(words))

def get_corpus_frequencies(corpus: list, language: str) -> dict:
    corpus_dict = {}
    for document in corpus:
        corpus_dict[document["docid"]] = get_document_vocabulary(preprocess_data(document["text"],language))
    return corpus_dict

def get_corpus_vocabulary(corpus_frequencies: dict) -> list:
    vocabulary = list()
    for doc_id in corpus_frequencies.keys():
        document_vocabulary = list(corpus_frequencies[doc_id].keys())
        vocabulary.extend(document_vocabulary)
    return list(set(vocabulary))

def term_frequency(word: str, document_frequency: dict) -> float:
    return document_frequency[word] / max(document_frequency.values())

def inverse_document_frequency(corpus_frequencies: dict, corpus_vocabulary: list) -> dict:

    document_count_per_word = collections.defaultdict(int)
    
    for doc_id, word_freq in corpus_frequencies.items():
        for word in word_freq.keys():
            document_count_per_word[word] += 1
    
    num_documents = len(corpus_frequencies)
    
    idf = {}
    for word in corpus_vocabulary:
        n_word = document_count_per_word[word]
        idf[word] = float(np.log(num_documents / n_word))
    
    return idf

def tf_idf(word: str, document_vocabulary: dict, idf: dict) -> float:
    return term_frequency(word, document_vocabulary)*idf[word]