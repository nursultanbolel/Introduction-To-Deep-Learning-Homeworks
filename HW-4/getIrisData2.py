# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:11:49 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
from sklearn.datasets import load_iris
import numpy as np

def getIrısData():
    iris=load_iris()
    X = iris.data
    
    trainingSet = np.zeros([120, 5])
    testSet = np.zeros([30, 5])
    y1 = np.zeros([120, 1])
    y2 = np.zeros([30, 1]) 
    
    
    trainingSet[0:40,0:4]=X[0:40,]
    trainingSet[40:80,0:4]=X[50:90,]
    trainingSet[80:120,0:4]=X[100:140,]
    
    trainingSet[0:120, 4]=1
    trainingSet = trainingSet.T
    
    testSet[0:10,0:4]=X[40:50,]
    testSet[10:20,0:4]=X[90:100,]
    testSet[20:30,0:4]=X[140:150,]
    
    testSet[0:30, 4]=1
    testSet = testSet.T
    
    for i in range(40):
        y1[i, :] = [0.1]
        y1[i + 40, :] = [0.5]
        y1[i + 80, :] = [1]
        
    for i in range(10):
        y2[i, :] = [0.1]
        y2[i + 10, :] = [0.5]
        y2[i + 20, :] = [1]
        
    
        
    return trainingSet,testSet,y1,y2