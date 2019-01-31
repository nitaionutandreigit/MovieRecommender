from math import ceil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings

from utils import (
                    weighted_rating, 
                    similar_genres, 
                    get_recommendations
                  ) 


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


class MovieRecommender(object):
    # define a constant quantile to determine the minimum rating to take into consideration
    QUANTILE = 0.85
    
    def __init__(self, movie_title):
        self.movie_title = movie_title

    def calculate_weight_rating(self, df):
        # Calculate the necessary variables for IMDB weight rating formula
        vote_counts = df[df['vote_count'].notnull()]['vote_count']
        vote_averages = df[df['vote_average'].notnull()]['vote_average']
        C = vote_averages.mean()
        m = vote_counts.quantile(self.QUANTILE) 
        movie_rating = df[df['title'] == self.movie_title]['vote_average']
        if movie_rating.empty:
            raise ValueError('Movie {} not found.'.format(self.movie_title))
        else:
            # Filter out movies with low vote count
            df_wr = df[(df['vote_count'] >= m)
                                     & (df['vote_average'] >= 0.85 * max(movie_rating))
                                     & (df['vote_count'].notnull()) 
                                     & (df['vote_average'].notnull())]
            # Vote count should be an integer
            df_wr['vote_count'] = df_wr['vote_count'].astype('int')
            # Vote average should be a float
            df_wr['vote_average'] = df_wr['vote_average'].astype('float')
            # Apply weight rating function to our dataframe
            df_wr['weighted_rating'] = df_wr.apply(weighted_rating, 
                args=(m, C,), axis=1)
            df_wr.sort_values(by=['weighted_rating'], ascending=False, inplace=True)
            df_wr.reset_index(inplace=True, drop=True)
            if 'rank' in df_wr.columns: 
                df_wr['rank'] += df_wr.index
            else:
                df_wr['rank'] = df_wr.index
        return df_wr

    def calculate_common_genres(self, df):
        # Format genres from JSON like string to list
        df['genres'] = df['genres'].apply(lambda x: [x['name'] 
            for x in eval(x)])
        # Select which movie you want to check genre for
        movie = df[df['title'] == self.movie_title]['genres']
        if movie.empty:
            raise ValueError('Movie {} not found.'.format(self.movie_title))
        else:
            # calculate how many common genres
            df['same_genres'] = df['genres'].apply(similar_genres, args=(movie,))
            # select the ones with the most common genres
            df = df[df['same_genres'] >= ceil(max(df['same_genres'] * 0.5))]
            # sort according to weighted rating and common genres
            df.sort_values(by=['same_genres'], ascending=False, inplace=True)
            df.reset_index(inplace=True, drop=True)
            if 'rank' in df.columns: 
                df['rank'] += df.index
            else:
                df['rank'] = df.index
        return df

    def calculate_tf_idf(self, df):
        # Retrieving index for movie
        movie_index = df[df['title'] == self.movie_title].index
        if movie_index.empty:
            raise ValueError('Movie {} not found.'.format(self.movie_title))
        else:
            #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
            tfidf = TfidfVectorizer(stop_words='english')
            # Construct the required TF-IDF matrix by fitting and transforming the data
            tfidf_matrix = tfidf.fit_transform(df['overview'])
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = cosine_sim[movie_index][0]
            df['sim_scores'] = sim_scores
            # Remove the movie from our dataframe
            df = df[df['title'] != self.movie_title]
            # Get rid of movies that have 0 similarity
            df = df[df['sim_scores'] >= max(df['sim_scores']) * 0.25]
            # sort according to common genres, weighted rating and similarity scores
            df.sort_values(by=['sim_scores'], ascending=False, inplace=True)
            df.reset_index(inplace=True, drop=True)
            if 'rank' in df.columns: 
                df['rank'] += df.index
            else:
                df['rank'] = df.index
        return df
