import pandas as pd
import json
from nltk.corpus import stopwords
import re


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


def preprocess_data(train_df, dev_df, test_df, corpus_df):

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

def save_cleaned_data(cleaned_data, output_dir='cleaned_data'):
    """ Saves cleaned DataFrames as JSON files in the specified directory. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for lang, df in cleaned_data.items():
        # Save each DataFrame to a separate JSON file
        output_path = os.path.join(output_dir, f'cleaned_data_{lang}.json')
        df.to_json(output_path, orient='records', lines=True)  # Save as JSON lines

# Testing block
if __name__ == '__main__':
    # Step 1: Load all the data (train, dev, test, and corpus)
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
    save_cleaned_data(cleaned_corpus_data, output_dir='cleaned_corpus_data')
