# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:48:28 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
import numpy as np

def sigm(X):
    h,w=X.shape
    O=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            X[i,j]=(-1)*X[i,j]
            O[i,j]= 1 / (1 + np.exp( X[i,j]))
            
    return O
