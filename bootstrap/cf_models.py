# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:13:38 2017

@author: tiwari
"""
import sys
import os
import requests
import urllib
import json
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
# from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from IPython.display import Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


# Headers = {'Accept': 'application/json'}
# payload = {'api_key': 'INSERT API KEY HERE'}
# response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
# response = json.loads(response.text)
# base_url = response['images']['base_url'] + 'w185'


def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

#top k collaborative
#
#userid_value=0
k=5
#r=5
ratings=train_data_matrix
similarity=calculate_similarity(train_data_matrix, kind='user', metric_type='cosine')
#i=0

def predict_topk(ratings, similarity,kind='user', k=50):
#    https://stackoverflow.com/questions/30332908/n-largest-values-in-each-row-of-ndarray
    masked_rating = np.ma.masked_array(ratings, mask=ratings==0)
    masked_mean_rating0=masked_rating.mean(axis=0)
    masked_mean_rating1=masked_rating.mean(axis=1)
    pred = np.zeros(ratings.shape)

    if kind == 'user':
        mean_rating = masked_mean_rating0
        #        mean_rating = np.mean(ratings,1)
        ratings_diff = (ratings - mean_rating[:, np.newaxis])
        
        np.mean(ratings_diff[0,:])
        
        
        top_k_similar_users = np.argsort(similarity, axis=1)[:,:-k - 1:-1]
        col_idx = np.arange(similarity.shape[0])[:,None]
        top_k_similarity=similarity[col_idx,top_k_similar_users]
        
        for i in xrange(top_k_similar_users.shape[0]):
            pred[i, :]= masked_mean_rating1[i] + (top_k_similarity[i,:].dot(ratings_diff[top_k_similar_users[i,:]])/np.sum(np.abs(top_k_similarity[i,:])))
    
    if kind == 'item':
        
        
    return pred



def predict_item_topk(ratings, similarity, k=50):
#    https://stackoverflow.com/questions/30332908/n-largest-values-in-each-row-of-ndarray

    masked_rating = np.ma.masked_array(ratings, mask=ratings==0)
    masked_mean_rating0=masked_rating.mean(axis=0)
    masked_mean_rating1=masked_rating.mean(axis=1)
    pred = np.zeros(ratings.shape)

    mean_rating = np.mean(ratings,1)
    ratings_diff = (ratings - mean_rating[:, np.newaxis])
    top_k_similar_users = np.argsort(similarity, axis=1)[:,:-k - 1:-1]
    col_idx = np.arange(similarity.shape[0])[:,None]
    top_k_similarity=similarity[col_idx,top_k_similar_users]
    pred = np.zeros(ratings.shape)
    
    for i in xrange(top_k_similar_users.shape[0]):
        pred[i, :]= mean_rating[i] + (top_k_similarity[i,:].dot(ratings_diff[top_k_similar_users[i,:]])/np.sum(np.abs(top_k_similarity[i,:])))

    return pred

    

def get_user_recommendations(ratings, similarity, userid_value, k=50, r=10):
#    pred = np.zeros(ratings.shape)
    mean_rating = np.mean(ratings,1)
    ratings_diff = (ratings - mean_rating[:, np.newaxis])
#    pred = np.zeros([1, ratings.shape[1]])
    top_k_users = np.argsort(similarity[:, userid_value])[:-k - 1:-1]
    pred=  mean_rating[userid_value] +  similarity[userid_value, :][top_k_users].dot(ratings_diff[top_k_users])/(np.sum(np.abs(similarity[userid_value,:][top_k_users])))
    mean_rating = np.mean(train_data_matrix,1)
    train_data_matrix_nobias = train_data_matrix - mean_rating[:,np.newaxis]
#    pred = np.zeros([1, ratings.shape[1]])
    top_k_users = np.argsort(similarity[:, userid_value])[:-k - 1:-1]
    pred=similarity[userid_value, :][top_k_users].dot(train_data_matrix[top_k_users])/(np.sum(np.abs(similarity[userid_value,:][top_k_users])))
    top_recomendation=np.argsort(pred)[:-r - 1:-1]
    #
#
#    for j in xrange(ratings.shape[1]):
#        if ratings[input_value, j] == 0:
#            pred[0, j] = similarity[input_value, :][top_k_users].dot(ratings[:, j][top_k_users]) / (np.sum(np.abs(similarity[input_value,:][top_k_users])))
#    top_recomendation = np.argsort(pred[0, :])[:-r - 1:-1]
#
    #    print similarity[input_data, :][top_k_users]
    #    print ratings[:, j][top_k_users]

    # print   pred[:, top_recomendation]
    return pred[np.newaxis,top_recomendation],top_recomendation 


#userid_value=0
#r=5
#
#a,b=get_user_recommendations(train_data_matrix,similarity,0,5,5)


def get_item_recommendations(ratings, similarity, input_value, k=5, r=10):
    pred = np.zeros([ratings.shape[0],1])
    calc1 = np.zeros([k, 3])
    top_k_items = np.argsort(similarity[input_value,:])[:-k - 1:-1]
    for j in xrange(ratings.shape[0]):
             pred[ j,0] = similarity[input_value,:][top_k_items].dot(ratings[j,:][top_k_items]) / (np.sum(np.abs(similarity[input_value,:][top_k_items])))

    calc1[:, 0] = input_value
    calc1[:, 1] = np.asarray(top_k_items)
    calc1[:, 2] = similarity[input_value, top_k_items]
    top_recomendation = np.argsort(pred[:,0])[:-r - 1:-1]

    return top_recomendation , pred[top_recomendation,:],calc1


def get_poster(r_tmdbid):
    headers = {'Accept': 'application/json'}
    payload = {'api_key': 'f16116de985abe2d0c9962bf168e36eb'}
    response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
    response = json.loads(response.text)
    base_url = response['images']['base_url'] + 'w185'
    #    r_tmdbid = 807
    request = 'https://api.themoviedb.org/3/movie/{0}?api_key=f16116de985abe2d0c9962bf168e36eb'.format(r_tmdbid)
    response = requests.get(request)
    response = json.loads(response.text)
    return Image(base_url + response['poster_path'])


def top_k_movies(similarity, mapper, movie_idx, k=6):
    return [mapper[x] for x in np.argsort(similarity[movie_idx, :])[:-k - 1:-1]]


def visualize(k_array,train_data_matrix,test_data_matrix,metric_type):

    user_train_mse = []
    user_test_mse  = []
    item_test_mse  = []
    item_train_mse = []

    item_similarity = calculate_similarity(train_data_matrix, kind='item', metric_type='cosine')
    user_similarity = calculate_similarity(train_data_matrix, kind='user', metric_type='cosine')

    for k in k_array:
       user_pred = predict_topk(train_data_matrix, user_similarity, kind='user', k=k)
       item_pred = predict_topk(train_data_matrix, item_similarity, kind='item', k=k)
       user_train_mse += [get_mse(user_pred, train_data_matrix)]
       user_test_mse += [get_mse(user_pred, test_data_matrix)]
       item_train_mse += [get_mse(item_pred, train_data_matrix)]
       item_test_mse += [get_mse(item_pred, test_data_matrix)]

    # %matplotlib inline
    sns.set()
    pal = sns.color_palette("Set2", 2)
    plt.figure(figsize=(8, 8))
    plt.plot(k_array, user_train_mse, c=pal[0], label='User-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, user_test_mse, c=pal[0], label='User-based test', linewidth=5)
    plt.plot(k_array, item_train_mse, c=pal[1], label='Item-based train', alpha=0.5, linewidth=5)
    plt.plot(k_array, item_test_mse, c=pal[1], label='Item-based test', linewidth=5)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('k', fontsize=30);
    plt.ylabel('MSE', fontsize=30);

    return (item_test_mse,user_test_mse)



def get_movies_data():

    os.getcwd()
    os.chdir('C:/python_code/recommendations')

    header = ['user_id', 'movie_id', 'rating', 'timestamp']
    df_ratings = pd.read_csv('data/ratings.csv', sep=',', names=header, header=0,
                             dtype={'user_id': np.int, 'item_id': np.int, 'rating': np.float, 'timestamp': np.int})
    header = ['movie_id', 'title', 'generes']
    movie_lookup = pd.read_csv('data/movies.csv', sep=',', names=header, header=0,
                               dtype={'movieid': np.int, 'title': np.str, 'generes': np.str})
    header = ['movie_id', 'imdbid', 'tmdbid']
    movie_links = pd.read_csv('data/links.csv', sep=',', names=header, header=0,
                              dtype={'movieid': np.int, 'imdbid': np.int, 'tmdbid': np.str})
    df_ratings['item_id'] = pd.factorize(df_ratings.movie_id)[0]
    movie_items = df_ratings[['item_id','movie_id']].copy().drop_duplicates()
    movie_lookup=movie_lookup.merge(movie_items, on ='movie_id')
    
    
    n_users = df_ratings.user_id.unique().shape[0]
    n_items = df_ratings.item_id.unique().shape[0]
    rating_train, rating_test = cv.train_test_split(df_ratings, test_size=0.2)

    train_data_matrix = np.zeros((n_users, n_items))
    test_data_matrix = np.zeros((n_users, n_items))

    train_data_matrix[rating_train.user_id.values-1,rating_train.item_id.values] = rating_train.rating.values
    test_data_matrix[rating_test.user_id.values-1,rating_test.item_id.values] = rating_test.rating.values
    
    writer = pd.ExcelWriter('data/df_ratings.xlsx', engine='xlsxwriter')
    df_ratings.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    
    conn = sqlite3.connect('data/df_ratings.db')
    conn.execute('''drop table rating_detail''')
    conn.execute('''create table rating_detail
                 (user_id text,
                  movie_id text,
                  rating numeric,
                  item_id numeric)
                 ''')

    df_ratings.to_sql("rating_detail", conn, if_exists="replace")
    conn.close()

#    for line in rating_train.itertuples():
#        train_data_matrix[line[1] - 1, line[5]] = line[3]
#
#
#    for line in rating_test.itertuples():
#        test_data_matrix[line[1] - 1, line[5]] = line[3]

    return (df_ratings, movie_lookup, movie_links, train_data_matrix, test_data_matrix)


def calculate_similarity(ratings, kind='item', metric_type='cosine'):
    if kind == 'item':
        sim = 1 - pairwise_distances(ratings.T, metric=metric_type)
    elif kind == 'user':
        sim = 1 - pairwise_distances(ratings, metric=metric_type)

    return sim


def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print movie_data[movie_data.movie_id == id].title.values[0]


def svd_similarity():

    os.chdir('C:\python_code\projects\movie_data')
    print (os.getcwd())
    
    
#start movie reco example 
    os.chdir('C:\python_code\projects\ml-1m')
    print os.getcwd()

    data = pd.io.parsers.read_csv('ratings.dat', 
        names=['user_id', 'movie_id', 'rating', 'time'],
        engine='python', delimiter='::')
    movie_data = pd.io.parsers.read_csv('movies.dat',
        names=['movie_id', 'title', 'genre'],
        engine='python', delimiter='::')

    ratings_mat = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values
    
    normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
    
    num_movies = ratings_mat.shape[0]
    
    a=normalised_mat.T
    
    A = normalised_mat / np.sqrt(num_movies - 1)
    
    VT.shape

    U,S,V = np.linalg.svd(A.T)
    
    b=a.dot(V.T)
    
    a=np.diag(S)
    
    normalised_mat = ratings_mat - np.matrix(np.mean(ratings_mat, 1)).T
    cov_mat = np.cov(normalised_mat)
    evals, evecs = np.linalg.eig(cov_mat)

    k = 50
    movie_id = 2 # Grab an id from movies.dat
    top_n = 10
    
    sliced = U[:, :k] # representative data
    indexes = top_cosine_similarity(sliced, movie_id, top_n)
    print_similar_movies(movie_data, movie_id, indexes)
    
    
    sliced2 = evecs[:, :k] # representative data
    top_indexes= top_cosine_similarity(sliced2, movie_id, top_n)
    print_similar_movies(movie_data, movie_id, top_indexes)
    
    #end section

    df_ratings, movie_lookup, movie_links, train_data_matrix, test_data_matrix = get_movies_data()

    item_similarity = calculate_similarity(train_data_matrix, kind='item', metric_type='cosine')
    v_top_reco,v_pred,v_calc=get_item_recommendations(train_data_matrix,item_similarity,0,5,10)

    train_data_matrix[0,v_top_reco]

#import scipy.sparse as sp
#from scipy.sparse.linalg import svds
#
#    u, s, vt = svds(train_data_matrix, k = 500)
#    
#    s_diag_matrix=np.diag(s)
#    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
#    a= get_mse(X_pred,train_data_matrix)
#    
#    print 'User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix))


    num_movies = train_data_matrix.shape[1]
    
    
    mean_train_data_matrix = np.asarray(np.mean(train_data_matrix, 1))
    mean_test_data_matrix = np.asarray(np.mean(test_data_matrix, 1))

    mean_center_train_data_matrix = train_data_matrix - mean_train_data_matrix[:,np.newaxis]
    mean_center_test_data_matrix =  test_data_matrix - mean_test_data_matrix[:,np.newaxis]
    


#    normalised_mat = train_data_matrix - np.asarray([(np.mean(train_data_matrix, 0))])

    mean_center_train_data_matrix = preprocessing.normalize(train_data_matrix)


    U, S, V = np.linalg.svd(mean_center_train_data_matrix)
    
    k = 50


    U_hat = U[:,:k]
    S_hat = np.diag(S)[:k,:k]
    V_hat = V[:k,:]
    
    train_data_matrix_hat = U_hat.dot(S_hat.dot(V_hat))

    train_data_matrix_hat = train_data_matrix_hat + mean_train_data_matrix[:,np.newaxis]

    rmse = get_mse(train_data_matrix_hat,train_data_matrix)
    print rmse
    
    b=mean_center_test_data_matrix.dot(V_hat.T)
    
    test_data_matrix_hat = b.dot(V_hat)
    test_data_matrix_hat = test_data_matrix_hat +mean_test_data_matrix[:,np.newaxis]
    
    
    rmse = get_mse(test_data_matrix_hat,test_data_matrix)
    print rmse
 
    
    k = 50
    movie_id = 0 # Grab an id from movies.dat
    top_n = 10
    
    sliced = V.T[:, :k] # representative data

    index = 0
    movie_row = sliced[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', sliced, sliced))
    
    similarity = np.dot(movie_row, sliced.T) / (magnitude[index] * magnitude)

    sort_indexes = np.argsort(-similarity)
    sort_indexes[:10]
    



    indexes = top_cosine_similarity(sliced, movie_id, top_n)
    print_similar_movies(movie_data, movie_id, indexes)
    
def pca_similarity():
    
#PCA    
    cov_mat = np.cov(normalised_mat.T)
    evals, evecs = np.linalg.eig(cov_mat)
    k = 50
    movie_id = 0 # Grab an id from movies.dat
    top_n = 10
    sliced = evecs[:, :k] # representative data
    top_indexes = top_cosine_similarity(sliced, movie_id, top_n)
    

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Helper function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print movie_data[movie_data.movie_id == id].title.values[0]

#    user_similarity = calculate_similarity(train_data_matrix,'user','cosine')

# c,t=get_recommendations(train_data_matrix,user_similarity,100,40,10)
# b=[get_poster(x[4]) for x in c ]
# display(*b)


# user_prediction = predict(train_data_matrix, user_similarity, type='user')
# get_mse(user_prediction,test_data_matrix)
#
# user_prediction = predict(train_data_matrix, user_similarity, type='user')
#
#
#
# user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
# user_similarity = 1-pairwise_distances(train_data_matrix,metric='cosine')
# user_prediction = predict(train_data_matrix, user_similarity, type='user')
#
# get_mse(user_prediction,test_data_matrix)
#
#
# item_similarity = cosine_similarity(train_data_matrix.T)
# user_prediction = predict(train_data_matrix.T, user_similarity, type='item')
# get_mse(user_prediction,test_data_matrix.T)
#
#
# get_mse(user_prediction_train,test_data_matrix)
# item_prediction_train = predict(train_data_matrix, item_similarity, type='item')
# get_mse(item_prediction_train,train_data_matrix)
# get_mse(item_prediction_train,test_data_matrix)
# get_mse(item_prediction_test,train_data)
#

def main():
    
    df_ratings, movie_lookup, movie_links, train_data_matrix, test_data_matrix = get_movies_data()
    user_similarity = calculate_similarity(train_data_matrix, kind='user', metric_type='correlation')
    
#    user_similarity = calculate_similarity(train_data_matrix, kind='item', metric_type='mahalanobis')

    pred_user_rating = predict_user_topk(train_data_matrix, user_similarity, k=50)
    get_mse(pred_user_rating, train_data_matrix)
    
    
    pred_user_rating = predict_user_topk(test_data_matrix, user_similarity, k=50)
    get_mse(pred_user_rating, test_data_matrix)
        
    a=pred_user_rating[pred_user_rating>6]
    
    v_top_reco,v_pred=get_user_recommendations(train_data_matrix,user_similarity,0,50,10)
 
 
   