import numpy as np
import pandas as pd
import pystan_utils
import matplotlib.pyplot as plt
def generate_fake_data(val_size=0.3,seed=42):
    n_group1 = 20
    n_group2 = 20
    n_movies_like_group1 = 20
    n_movies_like_group2 = 20
    userids_group1 = list(range(1,n_group1+1))
    userids_group2 = list(range(n_group1+1,n_group1 + n_group2+1))
    np.random.seed = seed
    train_set = []
    val_set = []
    for movie in range(1,n_movies_like_group1+1):
        for userid in userids_group1:
            record = [userid,movie,movie,1]
            if np.random.uniform()>val_size:
                never_sampled = True
                train_set.append(record)
            else:
                val_set.append(record)
        for userid in userids_group2:
            record = [userid,movie,movie,0]
            if np.random.uniform()>val_size:
                train_set.append(record)
            else:
                val_set.append(record)


    for movie in range(n_movies_like_group1+1, 
                       n_movies_like_group1 + n_movies_like_group2+1):
        for userid in userids_group2:
            record = [userid,movie,movie,1]
            if np.random.uniform()>val_size:
                train_set.append(record)
            else:
                val_set.append(record)
        for userid in userids_group1:
            record = [userid,movie,movie,0]
            if np.random.uniform()>val_size:
                train_set.append(record)
            else:
                val_set.append(record)


    train_set = pd.DataFrame(
        np.array(train_set), columns=['userId', 'movieId', 'movieIdNoHoles','like'])
    val_set = pd.DataFrame(
        np.array(val_set), columns=['userId', 'movieId', 'movieIdNoHoles','like'])
    return train_set, val_set

def generate_data_dict(train_set, val_set, n_traits=2, stars=False):
    num_movies = len(train_set.movieIdNoHoles.unique())
    num_users = len(train_set.userId.unique())
    data_dict =  {'num_movies': num_movies,
        'likes_obs': train_set['like'], 
        'num_traits': n_traits, 
        'num_users': num_users, 
        'num_likes': len(train_set), 
        'userId_obs': train_set['userId'],
        'movieId_obs': train_set['movieIdNoHoles'],
        'num_missing': len(val_set),
        'userId_missing': val_set['userId'],
        'movieId_missing': val_set['movieIdNoHoles']
    }

    if stars:
        data_dict['stars_obs']= train_set['rating']
    return data_dict, num_users, num_movies
def get_precision(predictions, val_set):
    true_labels = val_set['like']
    return 1 - sum(abs(predictions - true_labels))/len(true_labels)
    
def plot_low_variance_movies(fit,movies,num_movies,id_movie_dict, threshold=0.4):
    samples_var,means_var, stds_var, names_var=pystan_utils.vb_extract(fit)
    traits = []
    movieids = []
    for i in range(1,num_movies+1):
        #filter movies with poorly estimated traits
        if stds_var[f'trait[{i},1]']**2<threshold and stds_var[f'trait[{i},2]']**2<threshold:
            traits.append([means_var[f'trait[{i},1]'],means_var[f'trait[{i},2]']])
            movieids.append(id_movie_dict[i])

    traits = np.asarray(traits)

    plt.figure()
    ax = plt.scatter(traits[:,0], traits[:,1])
    print(f"number of movies with low variance posteriors = {len(movieids)}")
    for i, movieid in enumerate(movieids):
        try:
            plt.annotate(movies[movies.movieId==movieid].title.values[0], (traits[i,0], traits[i,1]))
        except:
            #Some of the movieId in ratings is not in the movie dataset
            pass
    


def get_NDCG(probabilitites, val_set, k=5):

    df = val_set.copy()
    df['probability'] = probabilitites
    
    #take k highest probability for each user
    df = df.sort_values('probability',ascending=False).groupby('userId').head(k)

    #Calculate the normalized discounted cumulative gain for each user
    result = 0.0
    n_users = len(df['userId'].unique())

    for userId in df['userId'].unique():
        
        ratings = df[df['userId']==userId]['rating']

        result += NDCG(ratings)
    #return the average over users
    return result/n_users
    
def DCG(r, k):
    """Score is discounted cumulative gain (dcg)

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
      
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def NDCG(r, k=5):
    """Score is normalized discounted cumulative gain (ndcg)

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
    
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = DCG(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return DCG(r, k) / dcg_max