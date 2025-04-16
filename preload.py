from sentence_transformers import SentenceTransformer

# This will trigger the model to download and be cached during build
SentenceTransformer('all-MiniLM-L6-v2')
