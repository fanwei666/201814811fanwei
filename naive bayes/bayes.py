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
def loadDataSet():
    path = "E://20news-18828//20news-18828" #文件夹目录
    files = os.listdir(path) #得到文件夹下的所有文件名称
    token = []
    words_freq = { }
    i = 1
    for file in files: #遍历文件夹
        file1 = file
        files1 = os.listdir(path+"//"+file1)
        for file in files1:
             file2 = file
             if not os.path.isdir(file2): #判断是否是文件夹，不是文件夹才打开
                 f = open(path+"//"+file1+"//"+file2,"rb") #打开文件
                 text = []
                 s = []
                 s.append(i)
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
                    if freq >= 2:
                        s.append(word)
                 token.append(s)
                 words_freq = { }
        i = i + 1
    return token

def train_test(dataSet):
    trainSet = []
    testSet = []
    documentNum = len(dataSet)
    trainSetNum = int(documentNum * 0.8)
    testSetNum = int(documentNum * 0.2)
    trainSet = random.sample(dataSet, trainSetNum)
    testSet = random.sample(dataSet, testSetNum)
    return trainSet,testSet

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    vocabList = []
    vocabList.append('class1010101010')
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    for element in vocabSet:
        vocabList.append(element)
    return vocabList

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * (len(vocabList))
    returnVec[0] = inputSet[0]
    for word in inputSet:
        if word in vocabList:
            if vocabList.index(word) != 0:
                returnVec[vocabList.index(word)] = 1
    return returnVec

def trainNB0(trainMatrix):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
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

def classifyNB(vec2Classify, pVect, pAb):
    p = [0.0] * 20
    for i in range(len(pVect)):
        p[i] = sum(vec2Classify * pVect[i]) + numpy.log(pAb[i])
    index = p.index(max(p))
    return index + 1

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * (len(vocabList))
    returnVec[0] = inputSet[0]
    for word in inputSet:
        if word in vocabList:
            if vocabList.index(word) != 0:
                returnVec[vocabList.index(word)] += 1
    return returnVec

def testBayes():
    #step1：加载数据集和类标号,分出测试集和训练集
    
    listOPosts = loadDataSet()
    trainSet,testSet = train_test(listOPosts)
    
    del listOPosts
    gc.collect()
    
    #step2：创建词库
    
    myVocabList = createVocabList(trainSet)

    
    # step3：计算每个样本在词库中的出现情况

    trainMat = []
    j = 0
    for postinDoc in trainSet:
        trainMat.append(bagOfWords2VecMN(myVocabList,postinDoc))
        del postinDoc
        gc.collect()
        j = j + 1
        print j
        
    #step4：调用第四步函数，计算条件概率

    del trainSet
    gc.collect()
    
    pVect,pAb = trainNB0(numpy.array(trainMat))

    del trainMat
    gc.collect()
    
    # step5

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