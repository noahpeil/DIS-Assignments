import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re
import os
import collections
import pickle
import numpy as np
import gc
from tqdm import tqdm
from embeddings_martin import load_model_and_tokenizer
from transformers import AutoTokenizer
import torch

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


def preprocess_data(text: str, language: str, do_stopwords: bool, do_lower: bool):

    if do_lower:
        text = text.lower()

    #text = re.sub(r'\W+', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = text.replace("\n"," ")

    if do_stopwords:
        try:
            stop_words = set(stopwords.words(language))
            text = ' '.join([word for word in text.split() if word not in stop_words])
        except:
            pass 
    return text


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
        with open(f"data/corpus_{language}.pkl","wb") as file:
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

def get_corpus_frequencies(corpus: list, language: str, do_stopwords: bool, do_lower: bool) -> dict: #Only lower capital letters at start of sentences ?
    corpus_dict = {}
    for document in corpus:
        corpus_dict[document["docid"]] = get_document_vocabulary(preprocess_data(document["text"],language, do_stopwords, do_lower).split(" "))
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

def batch_process_texts_token(texts, tokenizer, batch_size=1000):
    tokens_to_avoid = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=False, return_tensors='pt')
        corpus_frequency = []
        for doc in tokens["input_ids"]:
            unique_tokens, counts = torch.unique(doc, return_counts=True)
            corpus_frequency.append({unique_tokens[i].item():counts[i].item() for i in range(counts.shape[0]) if unique_tokens[i].item() not in tokens_to_avoid})
        # Split back into individual texts and yield
        yield corpus_frequency

def batch_process_texts_token_from_sentences(texts, tokenizer, language, do_stopwords, do_lower, batch_size=1000):
    tokens_to_avoid = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        sent_separator = " endofsentence "
        pattern = rf"\s*{re.escape(sent_separator.strip())}\s*"
        corpus_frequency = []
        for document in batch:
            sentences = re.split(r"(?<=[.?!]) +", document)
            processed_sentences = [item for item in re.split(pattern, preprocess_data(sent_separator.join(sentences), language, do_stopwords, do_lower)) if item]
            tokens = tokenizer(processed_sentences, padding=True, truncation=False, return_tensors='pt')
            unique_tokens, counts = torch.unique(tokens["input_ids"], return_counts=True)
            corpus_frequency.append({unique_tokens[i].item():counts[i].item() for i in range(counts.shape[0]) if unique_tokens[i].item() not in tokens_to_avoid})
        # Split back into individual texts and yield
        yield corpus_frequency

def corpus_token_frequency(corpus: list, tokenizer: AutoTokenizer, language: str, do_stopwords: bool = True, do_lower: bool = False) -> list[dict]:
    texts = [doc['text'] for doc in corpus]
    processed_texts = []
    for text in texts:
        processed_texts.append(preprocess_data(text, language, do_stopwords, do_lower))
    full_corpus_frequency = []
    for batch_corpus_frequency in tqdm(batch_process_texts_token(processed_texts, tokenizer)):
        full_corpus_frequency.extend(batch_corpus_frequency)
    return full_corpus_frequency

def corpus_token_frequency_from_sentences(corpus: list, tokenizer: AutoTokenizer, language: str, do_stopwords: bool = True, do_lower: bool = False) -> list[dict]:
    texts = [doc['text'] for doc in corpus]
    full_corpus_frequency = []
    for batch_corpus_frequency in tqdm(batch_process_texts_token_from_sentences(texts, tokenizer, language, do_stopwords, do_lower)):
        full_corpus_frequency.extend(batch_corpus_frequency)
    return full_corpus_frequency

def get_corpus_token_vocabulary(corpus_frequencies: list) -> list:
    vocabulary = list()
    for doc_freq in corpus_frequencies:
        document_vocabulary = list(doc_freq.keys())
        vocabulary.extend(document_vocabulary)
    return list(set(vocabulary))

def inverse_document_frequency_token(corpus_frequencies: list, corpus_vocabulary: list) -> dict:

    document_count_per_word = collections.defaultdict(int)
    
    for word_freq in corpus_frequencies:
        for word in word_freq.keys():
            document_count_per_word[word] += 1
    
    num_documents = len(corpus_frequencies)
    
    idf = {}
    for word in corpus_vocabulary:
        n_word = document_count_per_word[word]
        idf[word] = float(np.log(num_documents / n_word))

    return idf

def term_frequency(word: str, document_frequency: dict) -> float:
    return document_frequency[word] / max(document_frequency.values())

def tf_idf(word: str, document_frequency: dict, idf: dict) -> float:
    return term_frequency(word, document_frequency)*idf[word]

def batch_process_texts(texts, separator, language, do_stopwords, do_lower, batch_size=1000):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        combined_text = separator.join(batch)
        # Preprocess the combined text
        processed_text = preprocess_data(combined_text, language, do_stopwords, do_lower)
        # Split back into individual texts and yield
        yield processed_text.strip().split(separator)

def resize_sentences_selection(sentences: list[str], idf: dict, document_frequency: dict, language: str, do_stopwords: bool, do_lower: bool, selection_method: str, max_length: int = 512) -> str:
    if language != 'english':
        sent_separator = " endofsentence "
    else:
        sent_separator = " findephrase "
    processed_sentences = preprocess_data(sent_separator.join(sentences), language, do_stopwords, do_lower).split(sent_separator)
    #print(len(processed_sentences))
    processed_sentences_with_stopwords = preprocess_data(sent_separator.join(sentences), language, False, False).split(sent_separator)
    tf_idf_sentences = []
    len_sentences = []
    for sentence, sentence_with_stopwords in zip(processed_sentences, processed_sentences_with_stopwords):

    #for raw_sentence in sentences:
        #sentence = preprocess_data(raw_sentence, language, do_stopwords, do_lower)
        words = sentence.strip().split(" ") #Use a tokenizer ?
        if words != ['']:
            sentence_with_stopwords = preprocess_data(sentence, language, False, False)
            len_sentences.append(len(sentence_with_stopwords.strip().split(" ")))
            if selection_method == "mean":
                tf_idf_total = 0
                for word in words:
                    tf_idf_total += tf_idf(word, document_frequency, idf)
                tf_idf_sentences.append(tf_idf_total/len(words))

            elif selection_method == "exp_mean":
                tf_idf_total = 0
                for word in words:
                    tf_idf_total += np.exp(tf_idf(word, document_frequency, idf))
                tf_idf_sentences.append(tf_idf_total/len(words))

            elif selection_method == "sum":
                tf_idf_total = 0
                for word in words:
                    tf_idf_total += tf_idf(word, document_frequency, idf)
                tf_idf_sentences.append(tf_idf_total)

            elif selection_method == "max":
                tf_idf_words= []
                for word in words:
                    tf_idf_words.append(tf_idf(word, document_frequency, idf))
                tf_idf_sentences.append(max(tf_idf_words))
        else:
            tf_idf_sentences.append(-1)
            len_sentences.append(0)

    most_relevant = np.argsort(tf_idf_sentences)[::-1]
    ordered_len_sentences = np.array(len_sentences)[most_relevant]
    cum_sum_len = np.array([sum(ordered_len_sentences[:i]) for i in range(1,len(ordered_len_sentences)+1)])
    n_sentences_to_keep = (cum_sum_len <= max_length).sum().item()

    if n_sentences_to_keep != len(sentences):
        most_relevant_to_keep = most_relevant[:n_sentences_to_keep]
        resized_doc = " ".join([sentences[i] for i in range(len(sentences)) if i in most_relevant_to_keep])
    else:
        resized_doc =  " ".join(sentences)

    return resized_doc

def resize_sentences_selection_tokens(sentences: list[str], tokenizer: AutoTokenizer, idf_token: dict, document_frequency_token: dict, language: str, do_stopwords: bool, do_lower: bool, selection_method: str, max_length: int = 512) -> str:
    if language != 'english':
        sent_separator = " endofsentence "
    else:
        sent_separator = " findephrase "
    pattern = rf"\s*{re.escape(sent_separator.strip())}\s*"
    #processed_sentences = re.split(pattern,preprocess_data(sent_separator.join(sentences), language, True, True))
    processed_sentences = [item for item in re.split(pattern, preprocess_data(sent_separator.join(sentences), language, do_stopwords, do_lower)) if item]
    #print(len(processed_sentences))
    processed_sentences_with_stopwords = [item for item in re.split(pattern, preprocess_data(sent_separator.join(sentences), language, False, False)) if item]
    tf_idf_sentences = []
    len_sentences = []
    document_tokens = tokenizer(processed_sentences, padding=True, truncation=False, return_tensors='pt')["input_ids"]
    document_tokens_with_stopwords = tokenizer(processed_sentences_with_stopwords, padding=True, truncation=False, return_tensors='pt')["input_ids"]
    tokens_to_avoid = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
    for i in range(document_tokens.shape[0]):
        try:
            tokens_with_stopwords = document_tokens_with_stopwords[i]
            _, counts_with_stopwords = torch.unique(tokens_with_stopwords, return_counts=True)
            n_tokens_with_stopwords = counts_with_stopwords.sum().item()
            len_sentences.append(n_tokens_with_stopwords)
            tokens = document_tokens[i]
            unique_tokens, counts = torch.unique(tokens, return_counts=True)
            n_tokens = counts.sum().item()
            if selection_method == "mean":
                tf_idf_total = 0
                for j, token in enumerate(unique_tokens):
                    if token.item() not in tokens_to_avoid:
                        tf_idf_total += counts[j].item() * tf_idf(token.item(), document_frequency_token, idf_token)
                tf_idf_sentences.append(tf_idf_total/n_tokens)

            elif selection_method == "exp_mean":
                tf_idf_total = 0
                for j, token in enumerate(unique_tokens):
                    if token.item() not in tokens_to_avoid:
                        tf_idf_total += counts[j].item() * np.exp(tf_idf(token.item(), document_frequency_token, idf_token))
                tf_idf_sentences.append(tf_idf_total/n_tokens)

            elif selection_method == "sum":
                tf_idf_total = 0
                for j, token in enumerate(unique_tokens):
                    if token.item() not in tokens_to_avoid:
                        tf_idf_total += counts[j].item() * tf_idf(token.item(), document_frequency_token, idf_token)
                tf_idf_sentences.append(tf_idf_total)

            elif selection_method == "max":
                tf_idf_words= []
                for token in unique_tokens:
                    if token.item() not in tokens_to_avoid:
                        tf_idf_words.append(tf_idf(token.item(), document_frequency_token, idf_token))
                tf_idf_sentences.append(max(tf_idf_words))
        except Exception as e:
            print(i)
            print(processed_sentences[i])
            print(processed_sentences_with_stopwords[i])
            print(tokens)
            print(token)
            print(tokenizer(processed_sentences[i], padding=True, truncation=False, return_tensors='pt')['input_ids'])
            raise e

    most_relevant = np.argsort(tf_idf_sentences)[::-1]
    ordered_len_sentences = np.array(len_sentences)[most_relevant]
    cum_sum_len = np.array([sum(ordered_len_sentences[:i]) for i in range(1,len(ordered_len_sentences)+1)])
    n_sentences_to_keep = (cum_sum_len <= max_length).sum().item()

    if n_sentences_to_keep != len(sentences):
        most_relevant_to_keep = most_relevant[:n_sentences_to_keep]
        resized_doc = " ".join([sentences[i] for i in range(len(sentences)) if i in most_relevant_to_keep])
    else:
        resized_doc =  " ".join(sentences)

    return resized_doc

def select_most_relevant(texts: list[str], idf: dict, corpus_frequencies: dict, language: str, do_stopwords: bool, do_lower: bool, selection_method: str, max_length: int = 512) -> list[str]:
    relevant_texts = []
    has_resized_docs = 0
    for text, document_frequency in tqdm(zip(texts, corpus_frequencies.values())):
        if language not in ["arabic","korean"]:
            sentences = sent_tokenize(text, language)
        elif language == "arabic":
            sentences = re.split(r"(?<=[.!؟]) +", text)
        elif language == "korean":
            sentences = re.split(r"(?<=[.?!]) +", text)
        total_elements = sum([len(sentence.split(" ")) for sentence in sentences])
        if total_elements > max_length:
            resized_doc = resize_sentences_selection(sentences, idf, document_frequency, language, do_stopwords, do_lower, selection_method, max_length)
            relevant_texts.append(resized_doc)
            has_resized_docs += 1
        else:
            relevant_texts.append(text)
    if has_resized_docs:
        print(f"Resized {has_resized_docs} documents in {language} corpus")
    return relevant_texts

def select_most_relevant_tokens(texts: list[str], tokenizer: AutoTokenizer, idf_token: dict, corpus_frequencies_token: dict, language: str, do_stopwords: bool, do_lower: bool, selection_method: str, max_length: int = 512) -> list[str]:
    relevant_texts = []
    #has_resized_docs = 0
    for text, document_frequency_token in tqdm(zip(texts, corpus_frequencies_token)):
        if language not in ["arabic","korean"]:
            sentences = sent_tokenize(text, language)
        elif language == "arabic":
            sentences = re.split(r"(?<=[.!؟]) +", text)
        elif language == "korean":
            sentences = re.split(r"(?<=[.?!]) +", text)
        #total_elements = sum([len(sentence.split(" ")) for sentence in sentences])
        #if total_elements > max_length:
        resized_doc = resize_sentences_selection_tokens(sentences, tokenizer, idf_token, document_frequency_token, language, do_stopwords, do_lower, selection_method, max_length)
        relevant_texts.append(resized_doc)
        #has_resized_docs += 1
        #else:
        #    relevant_texts.append(text)
    #if has_resized_docs:
    #    print(f"Resized {has_resized_docs} documents in {language} corpus")
    return relevant_texts

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

    #JSON_PATH = "data/corpus.json"
    #with open(JSON_PATH,"r",encoding='utf-8') as file:
    #    data = json.load(file)

    #print("Creating corpus for each language")
    #create_language_corpus(data)
    #print("Corpus created for each language")
    gc.collect()
    #LANGUAGES = ["fr","it","es","de","ar","ko","en"]
    LANGUAGES = ["ko"]
    LANGUAGE_MAPPING = {"en":"english","fr":"french","de":"german","es":"spanish","ar":"arabic","ko":"korean","it":"italian"}
    for language in LANGUAGES:
        print(f"Loading corpus in {language}")
        with open(f"data/corpus_{language}.pkl","rb") as corpus_file:
            corpus = pickle.load(corpus_file)
        print(f"Corpus loaded in {language}")

        _, tokenizer = load_model_and_tokenizer(language)

        #corpus_frequencies = get_corpus_frequencies(corpus, LANGUAGE_MAPPING[language], True, True)
        if language != 'ko':
            corpus_frequencies = corpus_token_frequency(corpus, tokenizer, LANGUAGE_MAPPING[language], True, True)
        else:
            corpus_frequencies = corpus_token_frequency_from_sentences(corpus, tokenizer, LANGUAGE_MAPPING[language], True, True)
        with open(f"data/corpus_freq_{language}_token.pkl","wb") as freq_file:
            pickle.dump(corpus_frequencies, freq_file)
        print(f"Saved the {language} corpus frequency as data/corpus_freq_{language}_token.pkl")

        #corpus_vocabulary = get_corpus_vocabulary(corpus_frequencies)
        corpus_vocabulary = get_corpus_token_vocabulary(corpus_frequencies)
        with open(f"data/corpus_vocab_{language}_token.pkl","wb") as vocab_file:
            pickle.dump(corpus_vocabulary, vocab_file)
        print(f"Saved the {language} corpus vocabulary as data/corpus_vocab_{language}_token.pkl")

        #idf = inverse_document_frequency(corpus_frequencies, corpus_vocabulary)
        idf = inverse_document_frequency_token(corpus_frequencies, corpus_vocabulary)
        with open(f"data/corpus_idf_{language}_token.pkl","wb") as idf_file:
            pickle.dump(idf, idf_file)
        print(f"Saved the {language} corpus idf as data/corpus_idf_{language}_token.pkl")

        """with open(f"data/corpus_idf_{language}_token.pkl","rb") as idf_file:
            idf = pickle.load(idf_file)

        with open(f"data/corpus_freq_{language}_token.pkl","rb") as freq_file:
            corpus_frequencies = pickle.load(freq_file)"""

        processed_texts = list()
        texts = [document["text"] for document in corpus]

        relevant_texts = select_most_relevant_tokens(texts, tokenizer, idf, corpus_frequencies, LANGUAGE_MAPPING[language], True, True, "exp_mean", max_length=512)

        #if language != 'en':
        #    separator = "  endofdocument  "
        #else:
        #    separator = "  findedocument  "
        
        #for processed_batch in batch_process_texts(relevant_texts, separator, LANGUAGE_MAPPING[language], False, False, batch_size=1000):
        #    processed_texts += processed_batch

        processed_texts = [preprocess_data(text, language, False, False) for text in relevant_texts]

        if processed_texts:
            with open(f'data/processed_documents_{language}_exp_mean_relevancy.pkl','wb') as doc_file:
                pickle.dump(processed_texts, doc_file)
            print(f"Saved the {language} processed texts as data/processed_documents_{language}_exp_mean_relevancy.pkl")
        else:
            print(f"No data to save in {language}.")
