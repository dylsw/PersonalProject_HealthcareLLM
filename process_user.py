# handle_user_query.py


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the model (for user question embedding)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to handle user query
def handle_query(user_question):
    # Load precomputed embeddings and FAQ data
    embeddings_df = pd.read_csv("csv_data/faq_embeddings.csv")
    qns = pd.read_csv("csv_data/singhealth_test.csv")
    embeddings = embeddings_df.values

    # Embed the user question
    qn_embedding = model.encode([user_question])

    # Calculate cosine similarity and find the best match
    similarity_scores = cosine_similarity(qn_embedding, embeddings)[0]
    most_similar_index = np.argmax(similarity_scores)
    highest_similarity_score = similarity_scores[most_similar_index]

    # Set similarity threshold and get the best answer
    threshold = 0.25  # Adjust this threshold as needed
    answer = "Sorry, I do not understand your question."
    if highest_similarity_score > threshold:
        answer = qns['answer'].iloc[most_similar_index]
    
    #print(highest_similarity_score)

    return answer

# # Example usage
# sample_question = "is health hub better than health buddy?"
# print("User Question:", sample_question)
# print("Best Answer:", handle_query(sample_question))
