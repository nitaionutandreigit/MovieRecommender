'''
utils for movie recommendation model
'''
import pandas as pd


def weighted_rating(df, m, C):
    '''
    Function to calculate weighted rating
    based on imdb formula
    '''
    v = df['vote_count']
    R = df['vote_average']
    sum_m_v = m + v
    return (v / sum_m_v * R) + (m / sum_m_v * C)


def similar_genres(movie_genres, my_movie_genres):
    '''
    Function to calculate how many common genres 
    movies have between each other
    '''
    similarities = 0
    for series in my_movie_genres:
        for genre in series:
            if genre in movie_genres:
                similarities += 1
    return similarities


def add_movie(df, movie):
    df = pd.concat([df, movie])
    df.reset_index(inplace=True, drop=True)
    return df, df.tail(1).index


def calculate_rank(df, movie, sorting_column):
    df = df.loc[df['title'] != movie.title.values[0]]
    df.sort_values(by=sorting_column, ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)
    if 'rank' in df.columns: 
        df['rank'] += df.index
    else:
        df['rank'] = df.index
    return df.sort_values(by=['rank'])


def parse_cast_keywords(row):
    return ' '.join(set([x['name'].replace(' ', '') for x in eval(row)]))


def parse_crew(row):
    return ' '.join(set([x['name'].replace(' ', '') for x in eval(row) 
                if x['job'] in ('Director', 'Screenplay')]))


def concat_columns(df):
    return ' '.join(df['cast']) + ' ' + ' '.join(df['genres']) + ' ' + ' '.join(df['keywords'])
