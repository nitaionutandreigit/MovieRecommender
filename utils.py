'''
utils for movie recommendation model
'''


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


def get_recommendations(title, indices, cosine_sim):
    '''
    Function that takes in movie title as input and outputs most similar movies
    '''
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return movie_indices


def parse_cast_genre_keyword(row):
    return set([x['name'].replace(' ', '') for x in eval(row)])


def parse_crew(row):
    return set([x['name'].replace(' ', '') for x in eval(row) 
                if x['job'] in ('Director', 'Screenplay')])


def concat_columns(df):
    return ' '.join(df['cast']) + ' ' + ' '.join(df['genres']) + ' ' + ' '.join(df['keywords'])