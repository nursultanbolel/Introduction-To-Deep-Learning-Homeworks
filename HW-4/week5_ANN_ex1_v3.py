# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:57:49 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
import numpy as np
import sigm as S
import sigmDerivative as SG

input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input=input.T

groundTruth = np.array([[0], [1], [1], [0]])

bias = np.zeros((1, 3))
bias[0, 0:3] = [-1, -1, -1]

coeff = 0.7
iterations = 10000
#rand('state',sum(100*clock));
inputLength=2
hiddenN=2
outputN=1

w1=np.ones((hiddenN,inputLength+1))
w1[0,:]=0.1
w1[1,:]=0.2

w2=np.ones((outputN,hiddenN+1))
w2[0,:]=0.3

inputs=np.zeros((2, 1))
temp1=np.zeros((3, 1))
temp2=np.zeros((3, 1))
temp3=np.zeros((1, 2))

HL1=np.zeros((2, 1))
x3_1=np.zeros((3,3))
out=np.zeros((4,1))

for i in range(iterations):
    out = np.zeros((4, 1))
    for j in range(4):
        inputs[:,0]=input[:,j]
        #HL1 = w1*[-1; inputs]; 
        temp1[0,0]=-1
        temp1[1,0]=inputs[0,0]
        temp1[2,0]=inputs[1,0]
        
        HL1=np.dot(w1,temp1)
    
        HiddenLayerOutput1 = S.sigm(HL1)
        
        temp2[0,0]=-1
        temp2[1,0]=HiddenLayerOutput1[0,0]
        temp2[2,0]=HiddenLayerOutput1[1,0]   
        
        x3_1=np.dot(w2,temp2)
        out[j,0]=S.sigm(x3_1)
        
        delta3 = np.dot(SG.sigmDerivative(x3_1),groundTruth[j,0]-out[j,0])
        
        temp3=w2[:,1:3].T
        delta2 = np.dot(np.multiply(SG.sigmDerivative(HL1),temp3),delta3)
    
        w2=w2+np.multiply(np.multiply(temp1.T,coeff),delta3)
        
        w1=w1+np.multiply(coeff,np.dot(delta2,temp1.T))
        
#TEST CODE   
input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
input=input.T       
groundTruth = np.array([[0], [1], [1], [0]])
out=np.zeros((4,1))        
        
for j in range(4):
    inputs[:,0]=input[:,j]
    
    temp1[0,0]=-1
    temp1[1,0]=inputs[0,0]
    temp1[2,0]=inputs[1,0]
    
    HL1=np.dot(w1,temp1)
    HiddenLayerOutput1 = S.sigm(HL1)
    
    temp2[0,0]=-1
    temp2[1,0]=HiddenLayerOutput1[0,0]
    temp2[2,0]=HiddenLayerOutput1[1,0]   
        
    x3_1=np.dot(w2,temp2)
    out[j,0]=S.sigm(x3_1)