from math import ceil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings

from utils import (
                    add_movie,
                    weighted_rating, 
                    similar_genres,
                    calculate_rank,
                    parse_cast_keywords,
                    parse_crew
                  ) 


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


class MovieRecommender(object):
    # define a constant quantile to determine the minimum rating to take into consideration
    QUANTILE = 0.85
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

    def __init__(self, df, movie_title):
        self.movie = self.movie_exists(df, movie_title)
        self.df = self.filter_df(df)

    def movie_exists(self, df, movie_title):
        movie = df.loc[df['title'] == movie_title]
        if movie.empty:
            raise ValueError('Movie {} not found.'.format(self.movie['title']))
        return movie

    def filter_df(self, df):
        df = df.loc[df['title'] != self.movie.title.values[0]]
        min_vote_average = 0.9 * max(self.movie['vote_average'])
        min_vote_count = 0.9 * max(self.movie['vote_count'])
        print('Minimum vote average: {}, minimum vote count: {}.'.format(min_vote_average, min_vote_count))
        df = df[(df['vote_average'] >= min_vote_average) &
                     (df['vote_count'] >= min_vote_count)]
        df.reset_index(inplace=True, drop=True)
        return df

    def weight_rating(self):
        # import pdb; pdb.set_trace()
        print('Calculating weighted rating based on IMDB formula.')
        movie_rating = self.movie['vote_average']
        # Calculate the necessary variables for IMDB weight rating formula
        vote_counts = self.df['vote_count']
        vote_averages = self.df['vote_average']
        C = vote_averages.mean()
        m = vote_counts.quantile(self.QUANTILE)
        # Apply weight rating function to our dataframe
        self.df['weight_rating'] = self.df.apply(weighted_rating, 
            args=(m, C,), axis=1)
        # Calculate rank again by sorting by weighted rating
        self.df = calculate_rank(self.df, self.movie, 'weight_rating')

    def common_genres(self):
        print('Calculating the similar genres between movies.')
        self.df, movie_index = add_movie(self.df, self.movie)
        # Format genres from JSON like string to list
        self.df['genres'] = self.df['genres'].apply(lambda x: ' '.join([x['name'] for x in eval(x)]))
        count_matrix = self.cv.fit_transform(self.df['genres'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        sim_genres = cosine_sim[movie_index][0]
        self.df['sim_genres'] = sim_genres
        self.df = calculate_rank(self.df, self.movie,'sim_genres')

    def tf_idf_overview(self):
        print('Calculating the overview similarity score.')
        self.df, movie_index = add_movie(self.df, self.movie)
        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = self.tfidf.fit_transform(self.df['overview'])
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        # Get the pairwsie similarity scores of all movies with that movie
        sim_overview = cosine_sim[movie_index][0]
        self.df['sim_overview'] = sim_overview
        # sort according to common genres, weighted rating and similarity scores
        self.df = calculate_rank(self.df, self.movie, 'sim_overview')

    def tf_idf_cast_crew(self):
        print('Calculating the cast and crew similarity score.')
        self.df, movie_index = add_movie(self.df, self.movie)
        # Parse columns from JSON like string
        self.df['cast'] = self.df['cast'].apply(parse_cast_keywords)
        self.df['crew'] = self.df['crew'].apply(parse_crew)
        self.df['keywords'] = self.df['keywords'].apply(parse_cast_keywords)
        # Create new soup column from cast, crew and keywords
        self.df['soup'] = self.df['crew'] + ' ' + self.df['cast'] + ' ' + self.df['keywords']
        # Construct the required TF-IDF matrix by fitting and transforming the data
        count_matrix = self.cv.fit_transform(self.df['soup'])
        # Compute the cosine similarity matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        # Get the pairwsie similarity scores of all movies with that movie
        sim_cast = cosine_sim[movie_index][0]
        self.df['sim_cast'] = sim_cast
        # sort according to common genres, weighted rating and similarity scores
        self.df = calculate_rank(self.df, self.movie, 'sim_cast')
