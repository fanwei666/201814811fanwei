# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import nltk 
#import random
#import numpy
#nltk.download('stopwords')
#nltk.download('wordnet')
#import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
stopworddic = set(stopwords.words('english'))  
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
        
# 预处理数据
import os
path="E://20news-18828//20news-18828" #文件夹目录
text1=[]
files= os.listdir(path) #得到文件夹下的所有文件名称
#print files
for file in files: #遍历文件夹
    #path=path+"//"+file
    file1=file
    files1= os.listdir(path+"//"+file1)
    #print files1
    for file in files1:
         file2=file
         #print file1 
         if not os.path.isdir(file2): #判断是否是文件夹，不是文件夹才打开
             f = open(path+"//"+file1+"//"+file2,"rb") #打开文件
             #print f
             sentence=""
             s=[]
             text=[]
             mat=[]
             iter_f=iter(f)
             for line in iter_f:
                 #print line
                 temp=line.split(" ")
                 s.append(temp)
             lenth = len(s)
             j=0
             for single_list in s:
                 mat = [i for i in single_list if i not in stopworddic ]
                 #print mat
                 for word in mat:
                     try:
                         words=porter_stemmer.stem(word)
                         wordss=(wordnet_lemmatizer.lemmatize(words)).strip("\n,:.[]|>()?-")
                     except UnicodeDecodeError:
                         continue
                     if wordss != "\n" and wordss != '':
                         text.append(wordss)
                    
                 j = j + 1
                 if j == lenth - 1:
                     break;
                 #break;
             for word in text:
                sentence=sentence+word+" "
                #print sentence
             
             text1.append(sentence)
             #print text1
#print(text1)

#统计词频
print("1")
vectorizer=CountVectorizer()
print("2")
X=vectorizer.fit_transform(text1)
print("3")
word=vectorizer.get_feature_names()
print("4")
#print word
#print X.toarray()

#算TF-IDF
transformer = TfidfTransformer()
print("5")
#print transformer
#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
print("6")

        #print (element)
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
print("7")
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
    for j in range(len(word)):
        print word[j],weight[i][j]
        


#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
#print tfidf.toarray()

'''
#分出测试集与训练集
trainingSet=[]
testSet=[]
for x in range(len(mat)-1):
	if random.random() < 0.81:
		trainingSet.append(mat[x])
	else:
		testSet.append(mat[x])
print(len(trainingSet))
print(len(testSet))'''
