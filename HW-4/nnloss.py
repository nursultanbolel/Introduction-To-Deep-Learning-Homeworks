# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:24:31 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
import numpy as np

def nnloss(x, t, dzdy):
    instanceWeights = np.ones(x.shape)
    res = x - t 
    if dzdy==0 :
        return np.dot((1/2) * instanceWeights[:].T, pow(res,2))
    else:
        return res
