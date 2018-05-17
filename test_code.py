# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 06:49:59 2018

@author: User
"""

import os
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabaz_score
import numpy as np

data_path = 'C:\\Users\\User\\DekRobot\\Data\\'
range_n_clusters = [2, 250, 500, 750, 1000, 1500, 2500, 4000, 6000, 8000, 10000, 12000, 15000]



def read_file(data_path):
    file_number = 0
    file_to_work = list()
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            file_number += 1
            file_to_work.append(data_path + filename)
    return(file_to_work, file_number)


def get_unique_raws(file_to_work):
    total_data = set()
    for file in file_to_work:
        with open(file, 'r') as f:
            output = csv.reader(f)
            for raw in output:
                total_data.add('*'.join(raw))
    return total_data


def result_list_data(unique_raws_set):
    result_list = list()
    for raw in unique_raws_set:
        result_list.append(raw.split('*'))
    return result_list


def clean_third_axes(data_list):
    result_list = list()
    for I in data_list:
        if '-' in I[2]:
            total = ''
            for D in I[2].split('-'):
                if D:
                    total += D
            result_list.append([I[0], I[1], total])
        else:
            result_list.append(I)
    return result_list


def clean_third_axes_1(data_list):
    result_list = list()
    for I in data_list:
        if '-' in I[2]:
            continue
        else:
            result_list.append(list(map(int, I)))
    return result_list


def preprocessing_data(X):
#    transformer = FunctionTransformer(np.log)
    transformer = StandardScaler(copy=True, with_mean=True, with_std=True)
    return transformer.fit_transform(X)


def silhouette_score_count(clean_list, range_n_clusters):
    result = list()
    X = preprocessing_data(np.asarray(clean_list))
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,
                           n_init=5, max_iter=10, n_jobs=-1, random_state=17)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels, random_state=17)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        result.append((n_clusters, silhouette_avg))
    return result


def calinski_harabaz_score_count(clean_list, range_n_clusters):
    result = list()
    X = preprocessing_data(np.asarray(clean_list))
    X1 = X[:,:]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,
                           n_init=5, max_iter=10, n_jobs=-1, random_state=17)
        cluster_labels = clusterer.fit_predict(X1)
        calinski_harabaz_avg = calinski_harabaz_score(X1, cluster_labels)
        print("For n_clusters =", n_clusters,
              "calinski harabaz score is :", calinski_harabaz_avg)
        result.append((n_clusters, calinski_harabaz_avg))
    return result


def inertia_count(clean_list, range_n_clusters):
    result = list()
    X = preprocessing_data(np.asarray(clean_list))
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters,
                           n_init=5, max_iter=10, n_jobs=-1, random_state=17)
        clusterer.fit_predict(X)
        print("For n_clusters =", n_clusters,
              "Sum of squared distances of samples to their closest cluster center", clusterer.inertia_)
        result.append((n_clusters, clusterer.inertia_))
    return result


def afinity_count(clean_list):
    result_list = list()
    X = preprocessing_data(np.asarray(clean_list))
    X1 = X[:len(clean_list)//2, :]
    X2 = X[len(clean_list)//2:, :]
    data_list = [X1, X2]
    for X_half in data_list:
        af = AffinityPropagation(preference=-50).fit(X_half)
        cluster_centers_indices = af.cluster_centers_indices_
        n_clusters_ = len(cluster_centers_indices)
        labels = af.labels_
        silhouette_avg = silhouette_score(X_half, labels)
        print("For n_clusters =", n_clusters_,
              "The average silhouette_score is :", silhouette_avg)
        result_list.append((n_clusters_, silhouette_avg))
    return(n_clusters_, silhouette_avg)



if __name__ == '__main__':
    file_to_work, file_number = read_file(data_path)
    unique_raws_set = get_unique_raws(file_to_work)
    data_list = result_list_data(unique_raws_set)
    clean_list = clean_third_axes_1(data_list)
#    afinity_result = afinity_count(clean_list)
#    calinski_harabaz_result = calinski_harabaz_score_count(clean_list, range_n_clusters)
#    silhouette_result = silhouette_score_count(clean_list, range_n_clusters)
    inertia_result = inertia_count(clean_list, range_n_clusters)