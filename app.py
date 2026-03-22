import streamlit as st
import pickle
import pandas as pd

# ML imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pickle.load(open('movies.pkl', 'rb'))

new_df = load_data()


# ---------------- COMPUTE SIMILARITY ----------------
@st.cache_resource
def compute_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['tags']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(new_df)


# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies


# ---------------- UI ----------------
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox(
    "Select a movie",
    new_df['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    
    st.subheader("Top Recommendations:")
    for movie in recommendations:
        st.write(movie)
