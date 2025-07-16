import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "movies_with_posters.csv"))

# Lowercase genre column for consistent matching
df['Genre'] = df['Genre'].str.lower()

# Emotion to genre mapping
emotion_genre_map = {
    "happy": ["comedy", "romance"],
    "sad": ["drama", "biography"],
    "angry": ["action", "thriller"],
    "surprise": ["mystery", "sci-fi"],
    "fear": ["horror", "thriller"],
    "neutral": ["adventure", "drama", "action"]
}

# TF-IDF setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))

def recommend_movies(emotion, top_n=5):
    emotion = emotion.lower()
    if emotion not in emotion_genre_map:
        return pd.DataFrame()

    target_genres = emotion_genre_map[emotion]

    # Step 1: Genre-based filtering
    def genre_matches(row_genres):
        genres = [g.strip().lower() for g in row_genres.split(',')]
        return any(g in target_genres for g in genres)

    genre_filtered_df = df[df['Genre'].apply(genre_matches)]

    # Fallback 1: Try neutral emotion genres if current emotion gives no matches
    if genre_filtered_df.empty and emotion != 'neutral':
        neutral_genres = emotion_genre_map['neutral']
        genre_filtered_df = df[df['Genre'].apply(
            lambda row: any(g.strip().lower() in neutral_genres for g in row.split(',')))]
        print(f"[Fallback] No match for '{emotion}', used neutral genre fallback.")

    # Fallback 2: If still empty, return top-N based on global TF-IDF similarity
    if genre_filtered_df.empty:
        avg_vector = tfidf_matrix.mean(axis=0)
        avg_vector = np.asarray(avg_vector).reshape(1, -1)
        similarities = cosine_similarity(avg_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        print("[Fallback] No genre match, used global TF-IDF fallback.")
        return df.iloc[top_indices][['Movie_Title', 'Genre', 'Poster_URL', 'Description', 'YouTube_URL','Rating']]

    # Step 2: Description-based ranking within filtered subset
    genre_indices = genre_filtered_df.index.tolist()
    genre_vectors = tfidf_matrix[genre_indices]
    avg_vector = genre_vectors.mean(axis=0)
    avg_vector = np.asarray(avg_vector).reshape(1, -1)
    similarities = cosine_similarity(avg_vector, genre_vectors).flatten()
    top_filtered_indices = np.array(genre_indices)[similarities.argsort()[::-1][:top_n]]

    recommended = df.loc[top_filtered_indices]
    return recommended[['Movie_Title', 'Genre', 'Poster_URL', 'Description', 'YouTube_URL', 'Rating']]

# CLI test
if __name__ == "__main__":
    emotion = input("Enter emotion: ")
    print(recommend_movies(emotion))

