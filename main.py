import time
import numpy as np
import pandas as pd

from MovieRecommender import MovieRecommender


def main(movie_title):
    # read in the input csv files
    df_movies = pd.read_csv('movie_data/movies_metadata.csv', low_memory=False)
    df_cast = pd.read_csv('movie_data/cast.csv')
    df_crew = pd.read_csv('movie_data/crew.csv')
    df_keywords = pd.read_csv('movie_data/keywords.csv')

    # cast ids in credits and keywords to strings
    df_cast['id'] = df_cast['id'].astype('str')
    df_crew['id'] = df_crew['id'].astype('str')
    df_keywords['id'] = df_keywords['id'].astype('str')

    # merge cast, crew and keywords on the movies dataframe 
    df_movies = df_movies.merge(df_cast, on='id')
    df_movies = df_movies.merge(df_crew, on='id')
    df_movies = df_movies.merge(df_keywords, on='id')

    # Parse release date into new column only for year of release
    df_movies['year'] = pd.to_datetime(df_movies['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    # Select only the columns we are interested in
    df_movies = df_movies[['title', 'year', 'vote_count', 'vote_average', 
                   'popularity', 'overview', 'genres', 'cast', 'crew', 'keywords']]

    # Replace NaN with an empty string
    df_movies['overview'].fillna('', inplace=True)
    # fill nan with 0
    df_movies['vote_count'].fillna(0, inplace=True)
    df_movies['vote_average'].fillna(0, inplace=True)
    # Vote count should be an integer
    df_movies['vote_count'] = df_movies['vote_count'].astype('int')
    # Vote average should be a float
    df_movies['vote_average'] = df_movies['vote_average'].astype('float')

    mr = MovieRecommender(df_movies, movie_title)
    mr.weight_rating()
    mr.common_genres()
    mr.tf_idf_overview()
    mr.tf_idf_cast_crew()
    recommendations = mr.df.sort_values(by=['rank']).head(10)[['title', 'year', 'weight_rating', 'genres']]
    return recommendations

start = time.time()
print(main("Perfume: The Story of a Murderer"))
print('Results returned in {:.3f}'.format(time.time() - start))
