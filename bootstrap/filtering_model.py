# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:54:09 2018

@author: mtiw2
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


def process_data():

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
    rating_train, rating_test = cv.train_test_split(df_ratings, test_size=0.3)
    
    train_data_matrix = np.zeros((n_users, n_items))
    test_data_matrix = np.zeros((n_users, n_items))
    
    train_data_matrix[rating_train.user_id.values-1,rating_train.item_id.values] = rating_train.rating.values
    test_data_matrix[rating_test.user_id.values-1,rating_test.item_id.values] = rating_test.rating.values

    masked_rating = np.ma.masked_array(train_data_matrix, mask=train_data_matrix==0)
    user_similarity = 1 - pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric='cosine')


    mean_user_rating=masked_rating.mean(axis=1)
    user_ratings_diff = (masked_rating - mean_user_rating[:,np.newaxis])
    
    mean_item_rating=masked_rating.mean(axis=0)
    item_ratings_diff = (masked_rating - mean_item_rating[np.newaxis,:])

#start user based k=5 similar
#    k=50
    user_train_mse = []
    user_test_mse  = []
    item_test_mse  = []
    item_train_mse = []

    k_array=[5,10,50,100,150,200,250,300,350,400,450,500]
    
    for k in k_array:
        k=50
        #start user based 
        top_k_similar_users = np.argsort(user_similarity, axis=1)[:,:-k - 1:-1]
        col_idx = np.arange(user_similarity.shape[0])[:,None]
        top_k_user_similarity=user_similarity[col_idx,top_k_similar_users]
        user_pred = np.zeros(train_data_matrix.shape)
 
        for i in xrange(top_k_similar_users.shape[0]):
            user_pred[i, :]= mean_user_rating[i] + (top_k_user_similarity[i,:].dot(user_ratings_diff[top_k_similar_users[i,:]])/np.sum(np.abs(top_k_user_similarity[i,:])))

        user_test_mse  += [sqrt(mean_squared_error(user_pred[test_data_matrix.nonzero()].flatten(), test_data_matrix[test_data_matrix.nonzero()].flatten()))]
        user_train_mse += [sqrt(mean_squared_error(user_pred[train_data_matrix.nonzero()].flatten(), train_data_matrix[train_data_matrix.nonzero()].flatten()))]
        #end user based 

        #start item based 
        top_k_similar_items = np.argsort(item_similarity, axis=1)[:,:-k - 1:-1]
        col_idx = np.arange(item_similarity.shape[0])[:,None]
        top_k_item_similarity=item_similarity[col_idx,top_k_similar_items]
        item_pred = np.zeros(train_data_matrix.shape)
        

        for i in xrange(top_k_similar_items.shape[0]):
#            item_pred[:, i]= mean_item_rating[i] + (top_k_item_similarity[i,:].dot(item_ratings_diff.T[top_k_similar_items[i,:]])/np.sum(np.abs(top_k_item_similarity[i,:])))
            item_pred[:, i]= mean_item_rating[i] + (item_ratings_diff[:, top_k_similar_items[i,:]].dot(top_k_item_similarity[i,:]))/np.sum(np.abs(top_k_item_similarity[i,:]))
            
        item_test_mse+= [sqrt(mean_squared_error(item_pred[test_data_matrix.nonzero()].flatten(), test_data_matrix[test_data_matrix.nonzero()].flatten()))]
        item_train_mse+= [sqrt(mean_squared_error(item_pred[train_data_matrix.nonzero()].flatten(), train_data_matrix[train_data_matrix.nonzero()].flatten()))]

 #visualization
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


#using covariance matrix and eigen decomposition
    
    cov_mat = np.cov(item_ratings_diff.T)
    item_ratings_diff.shape
   
    
    evals, evecs = np.linalg.eig(cov_mat)
    k = 10
    sliced = evecs[:, :k] # representative data
    eigen_cosine_similarity= 1 - pairwise_distances(sliced, metric='cosine')
    top_n=20
    i=5
    print movie_lookup[movie_lookup.movie_id == i].title.values[0]
    recommended_movies=[ movie_lookup[movie_lookup.movie_id == y].title.values[0] for y in [movie_lookup[movie_lookup.item_id == x].movie_id.values[0] for x in np.argsort(eigen_cosine_similarity[i,:])[:-top_n-1:-1]]  ]
    print recommended_movies
    

#using SVD
    
    U, S, V = np.linalg.svd(item_ratings_diff)
    sliced = V.T[:, :k] # representative data
    svd_cosine_similarity= 1 - pairwise_distances(sliced, metric='cosine')
    print movie_lookup[movie_lookup.movie_id == i].title.values[0]
    recommended_movies=[ movie_lookup[movie_lookup.movie_id == y].title.values[0] for y in [movie_lookup[movie_lookup.item_id == x].movie_id.values[0] for x in np.argsort(svd_cosine_similarity[i,:])[:-top_n-1:-1]]  ]
    print recommended_movies
    

# reccomendations 

    k=10
    i=200
    print movie_lookup[movie_lookup.movie_id == i].title.values[0]
    recommended_movies=[ movie_lookup[movie_lookup.movie_id == y].title.values[0] for y in [movie_lookup[movie_lookup.item_id == x].movie_id.values[0] for x in np.argsort(item_similarity[i,:])[:-k-1:-1]]  ]
    print recommended_movies


    for x in recommended_movies:
        
        headers = {'Accept': 'application/json'}
        payload = {'api_key': 'f16116de985abe2d0c9962bf168e36eb'}
        response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
        response = json.loads(response.text)
        base_url = response['images']['base_url'] + 'w185'
        r_tmdbid = 807
        request = 'https://api.themoviedb.org/3/movie/{0}?api_key=f16116de985abe2d0c9962bf168e36eb'.format(r_tmdbid)
        response = requests.get(request)
        response = json.loads(response.text)
        Image(base_url + response['poster_path'])
    
    
   

def main():

    process_data()
    

