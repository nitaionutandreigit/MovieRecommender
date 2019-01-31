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
    df_credits['id'] = df_credits['id'].astype('str')
    df_keywords['id'] = df_keywords['id'].astype('str')

    # merge credits and keywords on the movies dataframe 
    df_movies = df_movies.merge(df_credits, on='id')
    df_movies = df_movies.merge(df_keywords, on='id')

    # Parse release date into new column only for year of release
    df_movies['year'] = pd.to_datetime(df_movies['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    # Select only the columns we are interested in
    df_movies = df_movies[['title', 'year', 'vote_count', 'vote_average', 
                   'popularity', 'overview', 'genres', 'cast', 'crew', 'keywords']]

    #Replace NaN with an empty string
    df_movies['overview'].fillna('', inplace=True)

    mr = MovieRecommender(movie_title)
    df_wr = mr.calculate_weight_rating(df_movies)
    df_genres = mr.calculate_common_genres(df_wr)
    df_tf_idf = mr.calculate_tf_idf(df_genres)
    print(df_tf_idf.sort_values(by=['rank']).head(10))


main('Life Is Beautiful')