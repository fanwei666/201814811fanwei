# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:55:12 2018

@author: fan
"""

import numpy as np
from sklearn import cluster
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from NMI import NMI

path="E://Tweets.txt"
f = open(path)
#data = []
token = []
label = []
for line in f.readlines():
    temp = line.replace("{","")
    temp = temp.replace("}","")
    temp = temp.replace("text","")
    temp = temp.replace("cluster","")
    temp = temp.split(",")
    for element in temp:
        #element.replace("\n","")
        if temp.index(element) % 2 == 0 :
            token.append(element.strip(' "":\n'))
        else:
            label.append(element.strip(' "":\n'))
length = len(set(label))
label = np.array(label)
    #data.append(token)
#word = np.array(data)
#print token

vectorizer=CountVectorizer(decode_error='ignore')#初始化对象
X=vectorizer.fit_transform(token)#词频

#word=vectorizer.get_feature_names()#词典中的单词

#算TF-IDF
transformer = TfidfTransformer()#初始化对象
tfidf = transformer.fit_transform(X)#将词频矩阵X统计成Ttf-idf值
#print tfidf
weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
#print weight

#print type(data)
#print data

labels = []
result = cluster.k_means(weight, length)
test_label = result[1]
for element in label:
    labels.append(int(element))
#print type(test_label)
print NMI(label,test_label)
    
#print result[1]
#print cluster.k_means(weight, length)