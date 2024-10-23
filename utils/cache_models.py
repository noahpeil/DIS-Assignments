from transformers import AutoModel, AutoTokenizer
import os


project_cache_dir = './huggingface_cache'

model_dict = {
    'en': 'bert-base-uncased',
    'fr': 'camembert-base',
    'de': 'bert-base-german-cased',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'it': 'dbmdz/bert-base-italian-cased',
    'ar': 'aubmindlab/bert-base-arabertv2',
    'ko': 'monologg/kobert'
}


def cache_model_and_tokenizer(language_code, model_name, cache_dir):
    print(f"Caching {language_code} model and tokenizer...")

    # Create language-specific subdirectory within the cache
    lang_cache_dir = os.path.join(cache_dir, language_code)
    os.makedirs(lang_cache_dir, exist_ok=True)

    # Download and cache the model and tokenizer
    _ = AutoTokenizer.from_pretrained(model_name, cache_dir=lang_cache_dir)
    _ = AutoModel.from_pretrained(model_name, cache_dir=lang_cache_dir)

    print(f"Model and tokenizer for {language_code} cached in {lang_cache_dir}.")


# Loop through the models for each language and cache them
for lang_code, model_name in model_dict.items():
    cache_model_and_tokenizer(lang_code, model_name, project_cache_dir)

print("All models and tokenizers cached.")
