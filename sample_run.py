"""This script contains the overall flow
"""

from preprocessing.preprocess import Preprocessing 
from clustering.dbscan import DBSCAN_clustering 
from search.search import searching


# Preprocess the data
prep = Preprocessing()
prep.preprocess()

# Clustering: only one model is demostrated here
clustering = DBSCAN_clustering()
time, labels = clustering.dbscan_optimal()
