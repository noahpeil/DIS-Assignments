import pandas as pd
import json
from nltk.corpus import stopwords
import re
import os
import collections
import pickle
import numpy as np
import gc

def load_data(train_path, dev_path, test_path, corpus_path):
    # Load the CSV files into pandas DataFrames
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Load corpus.json and convert it into a DataFrame
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    corpus_df = pd.DataFrame(corpus)

    return train_df, dev_df, test_df, corpus_df


def preprocess_data(text, language):

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    try:
        stop_words = set(stopwords.words(language))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    except:
        pass 
    return text

    # raise NotImplementedError


def split_data_by_language(df):
    """
    Splits data by each unique language in the dataset and stores them in a dictionary.

    Args:
        df (pd.DataFrame): DataFrame to split.

    Returns:
        dict: A dictionary where keys are language codes and values are DataFrames filtered by that language.
    """
    # Identify unique languages in the 'lang' column
    language_dfs = {lang: df[df['lang'] == lang] for lang in df['lang'].unique()}
    return language_dfs

def create_language_corpus(corpus: list) -> dict:
    language_corpus = {"en":[],"fr":[],"es":[],"de":[],"it":[],"ar":[],"ko":[]}
    for document in corpus:
        language_corpus[document["lang"]].append(document)
    for language in language_corpus.keys():
        with open(f"data/corpus_{language}.pkl","wb",encoding='utf-8') as file:
            pickle.dump(language_corpus[language],file)
    del language_corpus

def save_cleaned_data(cleaned_data, output_dir='cleaned_data'):
    """ Saves cleaned DataFrames as JSON files in the specified directory. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for lang, df in cleaned_data.items():
        # Save each DataFrame to a separate JSON file
        output_path = os.path.join(output_dir, f'cleaned_data_{lang}.json')
        df.to_json(output_path, orient='records', lines=True)  # Save as JSON lines

def get_document_vocabulary(words: str) -> dict:
    return dict(collections.Counter(words))

def get_corpus_frequencies(corpus: list, language: str) -> dict:
    corpus_dict = {}
    for document in corpus:
        corpus_dict[document["docid"]] = get_document_vocabulary(preprocess_data(document["text"].split(" "),language))
    return corpus_dict

def get_corpus_vocabulary(corpus_frequencies: dict) -> list:
    vocabulary = list()
    for doc_id in corpus_frequencies.keys():
        document_vocabulary = list(corpus_frequencies[doc_id].keys())
        vocabulary.extend(document_vocabulary)
    return list(set(vocabulary))

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

# Testing block
if __name__ == '__main__':

    """# Step 1: Load all the data (train, dev, test, and corpus)
    train_df, dev_df, test_df, corpus_df = load_data(
        'data/train.csv',
        'data/dev.csv',
        'data/test.csv',
        'data/corpus.json'
    )

    # Step 2: Split data by language
    train_dfs_by_language = split_data_by_language(train_df)
    dev_dfs_by_language = split_data_by_language(dev_df)
    test_dfs_by_language = split_data_by_language(test_df)
    corpus_dfs_by_language = split_data_by_language(corpus_df)

    # Step 3: Preprocess data for each language
    cleaned_train_data = preprocess_data(train_dfs_by_language)
    cleaned_dev_data = preprocess_data(dev_dfs_by_language)
    cleaned_test_data = preprocess_data(test_dfs_by_language)
    cleaned_corpus_data = preprocess_data(corpus_dfs_by_language)

    # Step 4: Save cleaned data as separate JSON files
    save_cleaned_data(cleaned_train_data, output_dir='cleaned_train_data')
    save_cleaned_data(cleaned_dev_data, output_dir='cleaned_dev_data')
    save_cleaned_data(cleaned_test_data, output_dir='cleaned_test_data')
    save_cleaned_data(cleaned_corpus_data, output_dir='cleaned_corpus_data')"""

    JSON_PATH = "data/corpus.json"
    with open(JSON_PATH,"r",encoding='utf-8') as file:
        data = json.load(file)

    print("Creating corpus for each language")
    create_language_corpus(data)
    print("Corpus created for each language")
    gc.collect()
    languages = ["en","fr","it","es","de","ar","ko"]
    for language in languages:
        print(f"Loading corpus in {language}")
        with open(f"data/corpus_{language}.pkl","rb",encoding='utf-8') as corpus_file:
            corpus = pickle.load(corpus_file)
        print(f"Corpus loaded in {language}")

        corpus_frequencies = get_corpus_frequencies(corpus, language)
        with open(f"data/corpus_freq_{language}.pkl","wb",encoding='utf-8') as freq_file:
            pickle.dump(corpus_frequencies, freq_file)
        print(f"Saved the {language} corpus frequency as data/corpus_freq_{language}.pkl")

        corpus_vocabulary = get_corpus_vocabulary(corpus_frequencies)
        with open(f"data/corpus_vocab_{language}.pkl","wb",encoding='utf-8') as vocab_file:
            pickle.dump(corpus_vocabulary, vocab_file)
        print(f"Saved the {language} corpus vocabulary as data/corpus_vocab_{language}.pkl")

        idf = inverse_document_frequency(corpus_frequencies, corpus_vocabulary)
        with open(f"data/corpus_idf_{language}.pkl","wb",encoding='utf-8') as idf_file:
            pickle.dump(idf, idf_file)
        print(f"Saved the {language} corpus idf as data/corpus_idf_{language}.pkl")
