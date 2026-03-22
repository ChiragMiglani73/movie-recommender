import streamlit as st
import pickle
import pandas as pd

new_df = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    return [new_df.iloc[i[0]].title for i in movies_list]


st.title("🎬 Movie Recommender")

selected_movie = st.selectbox(
    "Select a movie",
    new_df['title'].values
)

if st.button("Recommend"):
    for movie in recommend(selected_movie):
        st.write(movie)