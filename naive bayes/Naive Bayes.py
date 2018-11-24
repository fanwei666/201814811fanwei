# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:27:52 2018

@author: fan
"""

from numpy import *
import numpy
import os
import gc
import random

#获取所有文件的单词list，筛选单词以及缩小词典的大小
#得到的token list,每一行的第一个单词为该类的label
def loadDataSet():
    path = "E://20news-18828//20news-18828" #文件夹目录
    files = os.listdir(path) #得到文件夹下的所有文件名称
    token = []#用于存放word
    words_freq = { }#统计word的频率
    label = 1#label: 1--20
    for file in files: #遍历文件夹
        file1 = file
        files1 = os.listdir(path+"//"+file1)#遍历每个文件夹下的文件
        for file in files1:
             file2 = file
             if not os.path.isdir(file2): #判断是否是文件夹，不是文件夹才打开
                 f = open(path+"//"+file1+"//"+file2,"rb") #打开文件
                 text = []
                 s = []
                 s.append(label)
                 iter_f = iter(f)
                 for line in iter_f:
                     temp = line.split(" ")
                     if temp != '\n' and temp != '&' and len(temp) >3 and temp != ' ' and type(temp) != int and type(temp) != float:
                         text.append(temp)
                 for singlelist in text:
                     for word in singlelist:
                         word = word.strip("\n,:.[]|>""()?-!<>;+=_\/''")
                         if word in words_freq:
                             words_freq[word]+=1
                         else:
                             words_freq[word]=1
                 freq_words=[]
                 for word,freq in words_freq.items():
                     freq_words.append((freq,word))
                 freq_words.sort(reverse=True)
                 for freq,word in freq_words:
                    if freq >= 2:#只有当该word在文章中出现两次及以上，该word才会被放入字典中
                        s.append(word)
                 token.append(s)
                 words_freq = { }
        label +=  1
    return token

#分开train set和test set,大小比例为8：2
def train_test(dataSet):
    trainSet = []
    testSet = []
    from sklearn.cross_validation import train_test_split
    trainSet, testSet = train_test_split(dataSet, test_size = 0.2)
    return trainSet,testSet

#创建字典
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    vocabList = []
    vocabList.append('class1010101010') #添加一列，用于存放label
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    for element in vocabSet:
        vocabList.append(element)
    return vocabList

#统计字典里的每个word在每篇文章中出现的次数
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * (len(vocabList))
    returnVec[0] = inputSet[0]
    for word in inputSet:
        if word in vocabList:
            if vocabList.index(word) != 0:
                returnVec[vocabList.index(word)] += 1
    return returnVec

#训练train set
def trainNB0(trainMatrix):
    numTrainDocs = len(trainMatrix)#文章的数量
    numWords = len(trainMatrix[0])#字典的大小
    pSum = [0] * 20
    pAbusive = [0.0] * 20
    for i in range(numTrainDocs):
        pSum[trainMatrix[i][0]-1] += 1
    for i in range(len(pSum)):
        pAbusive[i] = pSum[i] / float(numTrainDocs) #计算某个类发生的概率
    pNum = [[1.0] * (numWords)] * 20
    pDenom = [2.0] * 20
    for i in range(numTrainDocs):
        index = trainMatrix[i][0] - 1
        trainMatrix[i][0] = 0
        for j in range(len(trainMatrix[i])):
            pNum[index] += trainMatrix[i]
        pDenom[index] += sum(trainMatrix[i])
    pVect = [0.0] * 20
    for i in range(20):
        pVect[i] = numpy.log(pNum[i]/pDenom[i])
    return pVect,pAbusive       #返回条件概率和类标签为1的概率

#对test set分类
def classifyNB(vec2Classify, pVect, pAb):
    p = [0.0] * 20#分别存放每个训练文章属于20个类的概率
    for i in range(len(pVect)):
        p[i] = sum(vec2Classify * pVect[i]) + numpy.log(pAb[i])
    index = p.index(max(p)) + 1#返回概率最大的类的label
    return index

#主函数
def testBayes():
    
    #step1：加载数据集和类label,分出测试集和训练集
    listOPosts = loadDataSet()
    trainSet,testSet = train_test(listOPosts)
    del listOPosts#释放内存空间
    gc.collect()
    
    #step2：创建字典
    myVocabList = createVocabList(trainSet)
    
    # step3：计算每个训练文章在字典中的出现情况
    trainMat = []
    j = 0
    for postinDoc in trainSet:
        trainMat.append(bagOfWords2VecMN(myVocabList,postinDoc))
        del postinDoc#释放内存空间
        gc.collect()
        j += 1
        print j
    del trainSet#释放内存空间
    gc.collect()    
    
    #step4：调用第四步函数，计算条件概率
    pVect,pAb = trainNB0(numpy.array(trainMat))
    del trainMat#释放内存空间
    gc.collect()
    
    # step5：测试每个测试文章，并统计最终正确率
    rightNum = 0.0
    wrongNum = 0.0
    for document in testSet:
        thisDoc = numpy.array(bagOfWords2VecMN(myVocabList, document))
        testClass = classifyNB(thisDoc,pVect,pAb)
        #print 'class',document[0],'test as',testClass
        if testClass == document[0]:
            rightNum += 1.0
        else:
            wrongNum += 1.0
    correctRate = 0.0
    correctRate = rightNum / (rightNum + wrongNum)
    print rightNum, wrongNum
    print 'correct rate: ', '%.4f' % correctRate

testBayes()