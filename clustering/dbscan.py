from numpy import genfromtxt
import numpy as np
from sklearn.cluster import DBSCAN
import time
import pandas as pd

from sklearn.metrics import silhouette_score
from random import sample 
import math
from itertools import combinations
import os
from pathlib import Path


class DBSCAN_clustering:

    #DEFAULT_PREPROCESSING_OUTPUT = "{}/preprocessing_output_0_7.csv".format(Path(os.getcwd()).parent)
    DEFAULT_PREPROCESSING_OUTPUT = "data/preprocessing_output.csv"

    """Optimal Parameters Obtained for DBSCAN
    """
    DEFAULT_OPTIMAL_EPS = 1.1


    DEFAULT_OUTPUT_FILE = "data/dbscan_output.csv"

    
    def dbscan_optimal(self, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT, optimal_eps = DEFAULT_OPTIMAL_EPS, output_file_name = DEFAULT_OUTPUT_FILE):

        df = genfromtxt(preprocess_file, delimiter = ",")
        dim = len(df)

        start_time = time.time()
        clustering = DBSCAN(eps = optimal_eps, min_samples = 2 * dim - 1).fit(df)
        end_time = time.time()

        output = pd.DataFrame(clustering.labels_, columns = ["label"])
        output.to_csv(output_file_name)
        return end_time - start_time, clustering.labels_

    def dbscan_tuning(self, eps_range, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT, optimal_eps = DEFAULT_OPTIMAL_EPS):
        
        df = genfromtxt(preprocess_file, delimiter = ",")
        dim = len(df)

        silhouette_avg = -100
        optimal_eps = 0
        for e in eps_range:
            try:
                clustering = DBSCAN(eps = e, min_samples = 2 * dim - 1).fit(df)
                n_cluster = len(np.unique(clustering.labels_))
                print("eps = {} has {} clusters".format(e, n_cluster))
                silhouette_avg = silhouette_score(df, clustering.labels_)
                if abs(silhouette_avg) > silhouette_avg:
                    optimal_eps = e
            except:
                pass
        
        return optimal_eps 

    def dbscan_analysis(self, preprocess_file = DEFAULT_PREPROCESSING_OUTPUT):
        df = genfromtxt(preprocess_file, delimiter = ",")
        overall_average = []
        lst = [x for x in range(len(df))]

        for count in range(10):
            s = sample(lst, 100)

            def dist(p1, p2):
                sqr_sum = 0
                for i in range(len(p1)):
                    sqr_sum += (p2[i] - p1[i]) ** 2
                return math.sqrt(sqr_sum)

            def average_dist(lst_points):
                count = 0
                distance_sum = 0
                for i in lst_points:
                    for j in lst_points:
                        count += 1
                        distance_sum += dist(i, j)
                return distance_sum/count

            lst_point = []
            for i in s:
                lst_point.append(df[i])
            overall_average.append(average_dist(lst_point))

        return sum(overall_average)/len(overall_average)


if __name__ == "__main__":

    algo = DBSCAN_clustering() 
    #print(algo.dbscan_analysis(preprocess_file = "subset_preprocess.csv"))
    time, labels = algo.dbscan_optimal(preprocess_file = "../preprocessing/subset_preprocess.csv", output_file_name = "dbscan_test.csv") 
    print(time)
