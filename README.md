# Movie Recommender System

This project is focused on building a content-based movie recommender system. The system uses various features of movies, such as genres, keywords, and cast, to recommend similar movies based on cosine similarity.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Loading](#data-loading)
3. [Data Preprocessing](#data-preprocessing)
    - [Merging Datasets](#merging-datasets)
    - [Handling Missing and Duplicate Values](#handling-missing-and-duplicate-values)
    - [Calculating Movie Age](#calculating-movie-age)
4. [Feature Engineering](#feature-engineering)
    - [Custom Functions for Data Conversion](#custom-functions-for-data-conversion)
    - [Combining Features into Tags](#combining-features-into-tags)
5. [Text Processing](#text-processing)
    - [Vectorization with CountVectorizer](#vectorization-with-count-vectorizer)
    - [Stemming with PorterStemmer](#stemming-with-porter-stemmer)
6. [Building the Recommender System](#building-the-recommender-system)
    - [Cosine Similarity Calculation](#cosine-similarity-calculation)
    - [Recommendation Function](#recommendation-function)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction

In this project, I build a movie recommender system using a content-based filtering approach. The goal is to recommend movies similar to a given movie based on certain features.

## Data Loading

Here I begin by loading the dataset from Kaggle. The dataset includes various features of movies that are used to generate recommendations.

```python
import pandas as pd
import numpy as np

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
```

## Data Preprocessing
### Merging Datasets
The datasets are merged on the title column to combine relevant information.
```python
movies = movies.merge(credits, on='title')
```
## Handling Missing and Duplicate Values
Remove any missing values and duplicates to clean the dataset.
```python
movies.dropna(inplace=True)
```
## Calculating Movie Age
A new column movie_age is created by calculating the age of each movie.
```python
movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month
movies['release_day'] = movies['release_date'].dt.day

# Calculate the age of the movie
movies['movie_age'] = (pd.to_datetime('today') - movies['release_date']).dt.days
```
## Feature Engineering
### Custom Functions for Data Conversion
Functions are created to convert genres, keywords, and cast columns into lists.
```python
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L
# fetch director from crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
```
## Combining Features into Tags
The genres, keywords, overview, and cast are concatenated into a tags column for easier processing.
```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew']
```
## Text Processing
### Vectorization with CountVectorizer
The tags are vectorized using CountVectorizer to transform text data into numerical features.
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```
## Stemming with PorterStemmer
Apply PorterStemmer to reduce words to their root forms for better similarity calculation.
```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)
```
## Building the Recommender System
### Cosine Similarity Calculation
Cosine similarity is used to calculate the similarity between different movies.
```python
from sklearn.metrics.pairwise import cosine_similarity
```
## Recommendation Function
A function is created to recommend the top 5 most similar movies based on cosine similarity.
```python
def recommendations(movie, weight_age=0.1):
    # Get the index of the movie that matches the title
    movie_index = new_df[new_df['title'] == movie].index[0]
    
    # Calculate similarity scores
    distances = similarity[movie_index]
    
    # Normalize movie_age (age should be normalized to have comparable influence)
    max_age = new_df['movie_age'].max()
    normalized_ages = new_df['movie_age'] / max_age
    
    # Combine similarity and movie age
    # You can adjust the `weight_age` to give more or less importance to movie age
    adjusted_distances = distances - weight_age * normalized_ages
    
    # Get a list of movies sorted by the combined score
    movie_list = sorted(list(enumerate(adjusted_distances)), reverse=True, 
                        key=lambda x: x[1])[1:6]
    
    # Print the titles of the recommended movies
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
```
## Conclusion
This notebook demonstrates the process of building a content-based movie recommender system using TMDB data. The project combines various movie features to suggest similar titles based on user preferences.

## References
- This project is a  built using data from [TMDB](https://www.themoviedb.org/). [KAGGLE](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
- Scikit-learn Documentation
- NLTK Documentation
