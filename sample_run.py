"""This script contains the overall flow
"""

from preprocessing.preprocess import Preprocessing 
from clustering.dbscan import DBSCAN_clustering 
from topic_modelling.topic_modelling import 
from search.search import searching


# Preprocess the data
prep = Preprocessing()
prep.preprocess()

# Clustering: only one model is demostrated here
clustering = DBSCAN_clustering()
time, labels = clustering.dbscan_optimal()


# Search
search_fun = searching()
print(search_fun.find_nearest("covid"))
