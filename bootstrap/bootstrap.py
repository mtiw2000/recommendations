# -*- coding: utf-8 -*-

"""bootstrap.bootstrap: provides entry point main()."""

__version__ = "0.3.0"

import sys
import csv
import os
import time
from cf_models import *
#from cf_models import get_movies_data
#from cf_models import calculate_similarity
from stuff import Stuff

# import matplotlib.pyplot as plt
# import seaborn as sns


# from recommendations import recomendation_engine

# def main():
#    print("Executing bootstrap version %s." % __version__)
#    print("List of argument strings: %s" % sys.argv[1:])
#    print("Stuff and Boo():\n%s\n%s" % (Stuff, Boo()))
#

def main():
    print("Executing bootstrap version %s." % __version__)
    total = len(sys.argv)

    print ("The total numbers of args passed to the script: %d " % total)
    print("List of argument strings: %s" % sys.argv[0:])
    print("Stuff and Boo():\n%s\n%s" % (Stuff, Boo()))
    #    Get the total number of args passed to the demo.py
    # Get the arguments list
    #    cmdargs = str(sys.argv)
    # Print it
    #    print ("Args list: %s " % cmdargs)

    os.chdir('C:\python_code\projects\movie_data')
    print os.getcwd()

    df_ratings, movie_lookup, movie_links, train_data_matrix, test_data_matrix = get_movies_data()

    item_similarity = calculate_similarity(train_data_matrix, kind='item', metric_type='cosine')
    user_similarity = calculate_similarity(train_data_matrix, kind='user', metric_type='cosine')

    item_similarity = calculate_similarity(train_data_matrix, kind='item', metric_type='correlation')
    user_similarity = calculate_similarity(train_data_matrix, kind='user', metric_type='correlation')

    v_top_reco,v_pred,v_calc=get_user_recommendations(train_data_matrix,user_similarity,0,5,10)
    v_top_reco,v_pred,v_calc=get_item_recommendations(train_data_matrix,item_similarity,0,5,10)


    #    recommend_data=recomendation_engine('testdata.csv','Y')
    print df_ratings[:5]

    start_time = time.time()
    print start_time

    # k_array = [5, 15, 30, 50, 75,100,125, 150,175,200,225,250,275,300,325,350,375,400]
    k_array = [5, 15, 30, 50, 75, 100]

    a,b=visualize(k_array, train_data_matrix, test_data_matrix,metric_type='correlation')

    #    print recommend_data


    end_time = time.time()

    print end_time - start_time



class Boo(Stuff):
    pass
