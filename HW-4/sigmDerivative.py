# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:53:23 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
import numpy as np

def sigmDerivative(X):
    h,w=X.shape
    Output=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            X[i,j]=(-1)*X[i,j]
            Output[i,j]= 1 / (1 + np.exp( X[i,j]))
            Output[i,j] = np.dot(Output[i,j],(1-Output[i,j]))
    return Output

