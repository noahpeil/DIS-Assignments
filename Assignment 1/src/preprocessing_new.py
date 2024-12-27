# This script will create separate, cleaned JSON files for each language in the specified output directory, making it ready for any language-specific or downstream processing.

import pandas as pd
import json
import re
import os
import pickle

# Import specific libraries for Arabic and Korean processing (optional)
try:
    from pyarabic.araby import sentence_tokenize as arabic_sentence_tokenize
except ImportError:
    arabic_sentence_tokenize = None  # Defaults to regex-based tokenization if not installed

try:
    import kss
except ImportError:
    kss = None  # Defaults to regex-based tokenization if not installed

# Load data
def load_corpus(path):
    """Load corpus from a JSON file and return as a list of dictionaries."""
    with open(path, 'r', encoding='utf-8') as file:
        corpus = json.load(file)
    return corpus

# Preprocess text
def preprocess_text(text, language):
    """Clean text: lowercase, remove unwanted characters, retain spaces and alphanumeric."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", ' ', text)  # Retain alphanumeric and spaces
    return text

# Split and save data by language as pickle
def split_and_save_by_language(corpus, output_dir):
    """Split corpus by language and save each language subset as a separate pickle file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Organize documents by language
    language_data = {}
    for document in corpus:
        lang = document["lang"]
        if lang not in language_data:
            language_data[lang] = []
        # Preprocess the document text before adding it
        document["text"] = preprocess_text(document["text"], lang)
        language_data[lang].append(document)

    # Save each language corpus to a separate pickle file
    for lang, docs in language_data.items():
        output_path = os.path.join(output_dir, f"corpus_{lang}.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(docs, file)
        print(f"Saved {lang} corpus with {len(docs)} documents to {output_path}")


# Calling it in the notebook

# # Load the corpus
# corpus_path = "data/corpus.json"
# corpus = load_corpus(corpus_path)

# # Preprocess, split by language, and save to output directory
# output_directory = "cleaned_corpora"
# split_and_save_by_language(corpus, output_directory)