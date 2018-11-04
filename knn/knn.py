# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:30:03 2018

@author: fan
"""

#########################################
# kNN: k Nearest Neighbors
 
# Input:      inX: vector to compare to existing dataset (1xN)
#             dataSet: size m data set of known vectors (NxM)
#             labels: data set labels (1xM vector)
#             k: number of neighbors to use for comparison 
            
# Output:     the most popular class label
#########################################
 
from numpy import *
import operator
import os
import VSM
 
# classify using kNN
def kNNClassify(newInput, dataSet, labels, k):#test[i],train_x
    numSamples = len(dataSet)  # shape[0] stands for the num of row
    print numSamples
    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    #print tile(newInput, (numSamples, 1))
    #print dataSet
    diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    #print diff
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    #print distance
    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)
   # print sortedDistIndices
    classCount = {} # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]
 
        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
 
    ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
 
    return maxIndex	
 
# convert image to vector
'''def  img2vector(filename):
 	rows = 32
 	cols = 32
 	imgVector = zeros((1, rows * cols)) 
 	fileIn = open(filename)
 	for row in xrange(rows):
 		lineStr = fileIn.readline()
 		for col in xrange(cols):
 			imgVector[0, row * 32 + col] = int(lineStr[col])
 
 	return imgVector'''
 
# load dataSet
def loadDataSet():
	## step 1: Getting training set
    print "---Getting training set..."
    '''dataSetDir = 'E:/Python/Machine Learning in Action/'
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set
    numSamples = len(trainingFileList)'''
    trainSet = []
    trainSet = VSM.trainSet()
    #print trainSet
    numSamples = len(trainSet)
   # train_x = zeros((numSamples,1024 ),dtype=str)
    train_y = []
    for i in range(numSamples):
        #filename = trainingFileList[i]
 
        # get train_x
        #train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename) 
        
        #train_x[i, :] = trainSet[i]
        #print train_x[i]
        # get label from file name such as "1_18.txt"
        #label = int(filename.split('_')[0]) # return 1
        train_y.append(i)
    #print trainSet
    ## step 2: Getting testing set
    print "---Getting testing set..."

    testSet = []
    testSet = VSM.testSet()
    #print testSet
    numSamples = len(testSet)
    #test_x = zeros((numSamples,1024 ),dtype=str)
    test_y = []
    for i in range(numSamples):
        #test_x[i, :] = testSet[i]
        test_y.append(i)
    #print train_x,test_x
    return trainSet, train_y, testSet, test_y

# test hand writing class
def testHandWritingClass():
	## step 1: load data
	print "step 1: load data..."
	train_x, train_y, test_x, test_y = loadDataSet()
 
	## step 2: training...
	print "step 2: training..."
	pass
 
	## step 3: testing
	print "step 3: testing..."
	numTestSamples = len(test_x)
	matchCount = 0
	for i in range(numTestSamples):
		predict = kNNClassify(test_x[i], train_x, train_y, 3)
		if predict == test_y[i]:
			matchCount += 1
	accuracy = float(matchCount) / numTestSamples
 
	## step 4: show the result
	print "step 4: show the result..."
	print 'The classify accuracy is: %.2f%%' % (accuracy * 100)

testHandWritingClass()