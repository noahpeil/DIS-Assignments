from transformers import AutoModel, AutoTokenizer
import torch
import os
import pickle


# Dictionary to map language codes to Hugging Face models
model_dict = {
    'en': 'bert-base-uncased',
    'fr': 'camembert-base',
    'de': 'bert-base-german-cased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'it': 'dbmdz/bert-base-italian-cased',
    'ar': 'aubmindlab/bert-base-arabertv2',
    'ko': 'monologg/kobert'
}


def load_model_and_tokenizer(language_code):
    """
    Load the appropriate model and tokenizer for the specified language.

    Args:
        language_code (str): Language code (e.g., 'en' for English).

    Returns:
        model: Pretrained Hugging Face model.
        tokenizer: Tokenizer corresponding to the model.
    """
    model_name = model_dict.get(language_code)
    if model_name is None:
        raise ValueError(f"Unsupported language code: {language_code}")

    # Use cached tokenizer and model
    cache_dir = os.path.join('./huggingface_cache', language_code)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    return model, tokenizer


def get_embeddings(texts, tokenizer, model):
    """
    Generate embeddings for a list of texts using the specified model and tokenizer.

    Args:
        texts (list of str): List of texts to embed.
        tokenizer: Tokenizer for the language-specific model.
        model: Pretrained Hugging Face model.

    Returns:
        torch.Tensor: Embeddings for the input texts.
    """
    # Tokenise the texts, using padding and truncation to handle different lengths
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Pass the inputs through the model
    with torch.no_grad():  # No gradient calculation needed for embedding generation
        outputs = model(**inputs)

    # Get the embeddings by averaging the hidden states across the sequence (mean pooling)
    # outputs.last_hidden_state: [batch_size, sequence_length, hidden_size]
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over the sequence dimension

    return embeddings


# TODO: Remove testing code below
if __name__ == '__main__':
    """# Example texts for each language
    texts_by_language = {
        'en': [
            "What is the capital of France?",
            "How to train a deep learning model?"
        ],
        'fr': [
            "Quelle est la capitale de la France ?",
            "Comment entraîner un modèle d'apprentissage profond ?"
        ],
        'de': [
            "Was ist die Hauptstadt von Frankreich?",
            "Wie trainiert man ein Deep-Learning-Modell?"
        ],
        'es': [
            "¿Cuál es la capital de Francia?",
            "¿Cómo entrenar un modelo de aprendizaje profundo?"
        ],
        'it': [
            "Qual è la capitale della Francia?",
            "Come addestrare un modello di deep learning?"
        ],
        'ar': [
            "ما هي عاصمة فرنسا؟",
            "كيف تدرب نموذج التعلم العميق؟"
        ],
        'ko': [
            "프랑스의 수도는 무엇입니까?",
            "딥러닝 모델을 훈련하는 방법은?"
        ]
    }
    
    # Iterate over each language and generate embeddings for sample texts
    for language_code, texts in texts_by_language.items():
        # Load model and tokenizer for the current language
        print(f"\nGenerating embeddings for {language_code} texts...")
        model, tokenizer = load_model_and_tokenizer(language_code)

        # Generate embeddings for the sample texts in this language
        embeddings = get_embeddings(texts, tokenizer, model)

        # Output the embeddings and their shape
        print(f"Texts: {texts}")
        print(f"Embeddings shape for {language_code}: {embeddings.shape}")  # Shape: [num_texts, hidden_size]
        torch.save(embeddings, f"data/tensor_{language_code}.pt")
    """

    languages = ['en','fr','de','es','it','ar','ko']

    for language in languages:
        with open(f'data/processed_documents_{language}.pkl','wb',encoding='utf-8') as doc_file:
            texts = pickle.load(doc_file)

        # Load model and tokenizer for the current language
        print(f"\nGenerating embeddings for {language} texts...")
        model, tokenizer = load_model_and_tokenizer(language)

        # Generate embeddings for the sample texts in this language
        embeddings = get_embeddings(texts, tokenizer, model)

        torch.save(embeddings, f"data/embeddings_tensor_{language}.pt")
        print(f"{language} embeddings saved as data/embeddings_tensor_{language}.pt")
        
