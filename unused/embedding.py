from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def embed_text(text):
    return model.encode(text)
