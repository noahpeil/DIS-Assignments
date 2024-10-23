import pandas as pd
import json


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
    # Nothing to do yet, might change later
    raise NotImplementedError


def split_data_by_language(df, language):
    # Filter the DataFrame to include only rows with the specified language
    return df[df['lang'] == language]


# TODO: remove testing code below
# For testing purposes
if __name__ == '__main__':
    # Step 1: Load all the data (train, dev, test, and corpus)
    train_df, dev_df, test_df, corpus_df = load_data(
        'data/train.csv',
        'data/dev.csv',
        'data/test.csv',
        'data/corpus.json'
    )

    '''
    # Step 2: Preprocess the data
    # No need to rename columns, 'lang' is already present in all DataFrames
    train_df, dev_df, test_df, corpus_df = preprocess_data(train_df, dev_df, test_df, corpus_df)
    '''

    # Step 3: Split the data by language (e.g., for English 'en')
    train_df_en = split_data_by_language(train_df, 'en')
    dev_df_en = split_data_by_language(dev_df, 'en')
    test_df_en = split_data_by_language(test_df, 'en')
    corpus_df_en = split_data_by_language(corpus_df, 'en')

    # Now, you can proceed with embedding generation, model training, etc.
    print(train_df_en.head(5))
