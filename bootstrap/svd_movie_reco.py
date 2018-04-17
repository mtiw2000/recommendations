# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:13:13 2018

@author: mtiw2
"""

import numpy as np
import pandas as pd
import sys
import os


def svd_process():

    os.getcwd()
    os.chdir('C:/python_code/recommendations')


    data = pd.io.parsers.read_csv('data/ml-1m/ratings.dat',names=['user_id', 'movie_id', 'rating', 'time'], engine='python', delimiter='::')
    movie_data = pd.io.parsers.read_csv('data/ml-1m/movies.dat', names=['movie_id', 'title', 'genre'],  engine='python', delimiter='::')


    ratings_mat = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),  dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values
    
    movie_rating_mean = np.asarray(np.mean(ratings_mat, 1))
    
    normalised_mat = ratings_mat - movie_rating_mean[:,np.newaxis]


    cov_mat = np.cov(normalised_mat)
    evals, evecs = np.linalg.eig(cov_mat)


k = 50
movie_id = 1 # Grab an id from movies.dat
top_n = 10

sliced = evecs[:, :k] # representative data
top_indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, top_indexes)


#    np.sum(normalised_mat[1,:])
    np.std(A[:,0])
    
    
    num_movies = movie_data.movie_id.unique().shape[0]
    
    A = normalised_mat/np.sqrt(num_movies - 1)
    
    U,S,V = np.linalg.svd(A)
    