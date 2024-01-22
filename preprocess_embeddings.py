# preprocess_embeddings.py


import pandas as pd
from sentence_transformers import SentenceTransformer

def handle_reload():
# Load questions
    qns = pd.read_csv("csv_data/singhealth_test.csv")

    # Generate embeddings for FAQ questions
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = [q for q in qns['question']]
    embeddings = model.encode(sentences)

    # Save embeddings to a CSV file
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv("csv_data/faq_embeddings.csv", index=False)

    print("Embeddings saved to 'csv_data/faq_embeddings.csv'")
