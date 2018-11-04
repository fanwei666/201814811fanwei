﻿# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 
import nltk 
import random
import os
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import *
import numpy
def VSM():
    stopworddic = set(stopwords.words('english'))  
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
            
    # 预处理数据
    path="20news-18828" #文件夹目录
    text1 = []
    files = os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历 20news-18828 下的文件夹
        file1 = file
        files1 = os.listdir(path+"//"+file1)
        for file in files1: #遍历 每个子文件夹 下的文件
            file2 = file
            if not os.path.isdir(file2): #判断是否是文件夹，不是文件夹才打开
                f = open(path+"//"+file1+"//"+file2,"rb") #打开文件
                sentence = ""
                s = []
                text = []
                mat = []
                iter_f = iter(f)#创建迭代器
                for line in iter_f:#按行读取文件
                    temp = line.split(" ")#将每行的句子按照‘ '分词
                    s.append(temp)#将每个单词存入list
                f.close()
                length = len(s)#用于判断循环结束
                j = 0
                for single_list in s:#s为多维list
                    mat = [i for i in single_list if i not in stopworddic ]#除去stopwords
                    for word in mat:
                        try:
                            words = porter_stemmer.stem(word)#去掉单词后的's',stem
                            wordss = wordnet_lemmatizer.lemmatize(words)#将单词转化为原始形式
                            wordsss = wordss.strip("\n,:.[]|>()?-")#去掉单词前后的符号
                        except UnicodeDecodeError:
                            continue
                        if wordsss != "\n" and wordss != '' and type(wordss) != int and len(wordss) > 3:#去掉换行和空
                            text.append(wordsss)
                    j = j + 1
                    if j == length - 1:
                        break;
                for word in text:
                    sentence=sentence+word+" "
                text1.append(sentence)
                 
    #分出测试集与训练集            
    '''trainingSet=[]
    testSet=[]
    for x in range(len(text1)-1):
    	if random.random() < 0.81:
    		trainingSet.append(text1[x])
    	else:
    		testSet.append(text1[x])'''
    #print(len(trainingSet))
    #print(len(testSet))
    
    #统计词频
    vectorizer=CountVectorizer()#初始化对象
    X=vectorizer.fit_transform(text1)#词频
    #print X.toarray()
    #tf = X.toarray()
    
    word=vectorizer.get_feature_names()#词典中的单词
    
    #算TF-IDF
    transformer = TfidfTransformer()#初始化对象
    tfidf = transformer.fit_transform(X)#将词频矩阵X统计成Ttf-idf值
    weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight[1])):
        if (weight[:,i]==0).all():
            weight = numpy.delete(weight, i, axis=1)
    #for i in range(len(weight)):
     #   for j in range(len(weight[i])):
      #      if weight[i][j] == 0:
       #         for k in range(len(weight)):
        #            weight.remove(weight[k][j])
    #print weight
    return weight
    #将结果存入txt文件
    '''f = open('C:\Users\fan\.spyder\VSM.txt', 'w')'''
def trainSet():
    weight = VSM()
    trainSet_o = [0]
    trainSet = [trainSet_o * len(weight)]
    #print trainSet
    #trainSet.append(trainSet_o)
    '''for i in range(len(weight)):
        print weight[i]'''
    #trainSet[length][]
    #print weight[1]
    print len(weight)
    for i in range(len(weight)):#每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        if i < (len(weight)) * 0.5:
            #print len(weight[i])
            weight_new = [weight[i]]
            trainSet = numpy.row_stack((trainSet, weight_new))
            #print weight_new
            #numpy.insert(trainSet,0,values=weight_new,axis=0)
    #print (trainSet)
    return trainSet

def testSet():
    weight = VSM()
    testSet_o = [0]
    testSet = [testSet_o * len(weight)]
    #print trainSet
    #trainSet.append(trainSet_o)
    '''for i in range(len(weight)):
        print weight[i]'''
    #trainSet[length][]
    #print weight[1]
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        if i > (len(weight)) * 0.8:
            #print len(weight[i])
            weight_new = [weight[i]]
            #print weight_new
            testSet = numpy.row_stack((testSet, weight_new))
            #numpy.append(testSet,weight_new,axis=0)

    #print testSet
    return testSet
        #f.write('-------this is: ' + str(i) + ' class tf-idf weight of one file------\n')
        #for j in range(len(word)):
            #if weight[i][j] != 0:
                #f.write(str(word[j]) + ' ' + str(weight[i][j]) + '\n')
    #f.close()
#trainSet()