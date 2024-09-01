import streamlit as st
import pickle
import pandas as pd
import requests

# Load the movie dictionary and similarity matrix
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open("similarity.pkl", "rb"))

# Fetch poster images
def fetch_poster_images(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=904656d2922662cef3d1cd7f8bc9a158".format(movie_id))
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image+Available"  # Placeholder if no image is found



# Recommendation function
def get_recommendations(movie, weight_age=0.1):
    # Get the index of the movie that matches the title
    movie_index = movies[movies['title'] == movie].index[0]
    
    # Calculate similarity scores
    distances = similarity[movie_index]
    
    # Normalize movie_age (age should be normalized to have comparable influence)
    max_age = movies['movie_age'].max()
    normalized_ages = movies['movie_age'] / max_age
    
    # Combine similarity and movie age
    adjusted_distances = distances - weight_age * normalized_ages
    
    # Get a list of movies sorted by the combined score
    movie_list = sorted(list(enumerate(adjusted_distances)), reverse=True, 
                        key=lambda x: x[1])[1:6]
    
    # Create a list of recommended movie titles
    recommended_movies = []
    recommended_movies_poster = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster images
        recommended_movies_poster.append(fetch_poster_images(movie_id))  # TODO: Add functionality to fetch poster images from the API))  # TODO: Add functionality to fetch poster images from the API
        

    return recommended_movies, recommended_movies_poster

# Add title and subtitle
st.title('Movie Recommender System')

# Dropdown for selecting a movie
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# Button to get recommendations
if st.button("Recommend"):
    recommended_movie_names, recommended_movie_posters = get_recommendations(selected_movie)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])