# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:52:00 2018

@author: fan
"""

import json
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering,AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

def loadData():
    DataDict = []
    Data = []
    DataLabels = []
    file = open("E:\\Tweets.txt",'r')
    for line in file.readlines():
        lines = json.loads(line)
        DataDict.append(lines)
    for line in DataDict:
        Data.append(line['text'])
        DataLabels.append(line['cluster'])
    tfidf = TfidfTransformer().fit_transform(CountVectorizer(decode_error='ignore',stop_words='english').fit_transform(Data))
    Data = tfidf.toarray()
    return Data,DataLabels

def Kmeans(Data,DataLabels):
    n_cluster_number = len(set(DataLabels))
    print("==============KMeans==============")
    DataCluster = KMeans(n_clusters=n_cluster_number, random_state=10 ).fit(Data)
    printResult(DataCluster.labels_,DataLabels)

def affinityPropagation(Data,DataLabels):
    print("=======AffinityPropagation========")
    clustering = AffinityPropagation().fit(Data)
    printResult(clustering.labels_,DataLabels)

def meanShift(Data,DataLabels):
    print("===========MeanShift=============")
    clustering = MeanShift().fit(Data)
    printResult(clustering.labels_,DataLabels)

def WardHierarchicalClustering(Data,DataLabels):
    n_cluster_number = len(set(DataLabels))
    print("====WardHierarchicalClustering====")
    clustering = AgglomerativeClustering(n_clusters=n_cluster_number,linkage='ward').fit(Data)
    printResult(clustering.labels_,DataLabels)

def spectralClustering(Data,DataLabels):
    n_cluster_number = len(set(DataLabels))
    print("=======SpectralClustering=========")
    DataCluster = SpectralClustering(n_clusters=n_cluster_number).fit(Data)
    printResult(DataCluster.labels_,DataLabels)

def agglomerativeClustering(Data,DataLabels):
    n_cluster_number = len(set(DataLabels))
    print("=====AgglomerativeClustering=====")
    DataCluster = AgglomerativeClustering(n_clusters=n_cluster_number).fit(Data)
    printResult(DataCluster.labels_,DataLabels)

def dBSCAN(Data,DataLabels):
    print("============DBSCAN===============")
    clustering = DBSCAN(eps=1.13).fit(Data)
    printResult(clustering.labels_,DataLabels)

def gaussianMixture(Data,DataLabels):
    n_cluster_number = len(set(DataLabels))
    print("=========GaussianMixture=========")
    GM = GaussianMixture(n_components=n_cluster_number,covariance_type='diag').fit(Data)
    clustering = GM.predict(Data)
    printResult(clustering,DataLabels)

def printResult(DataCluster,DataLabels):
    print("NMI: %s" % (metrics.normalized_mutual_info_score(DataLabels, DataCluster)))
    
if __name__ == '__main__':
    Data, DataLabels = loadData()
    
    Kmeans(Data,DataLabels)
    affinityPropagation(Data,DataLabels)
    meanShift(Data,DataLabels)
    WardHierarchicalClustering(Data,DataLabels)
    spectralClustering(Data,DataLabels)
    agglomerativeClustering(Data,DataLabels)
    dBSCAN(Data,DataLabels)
    gaussianMixture(Data,DataLabels)