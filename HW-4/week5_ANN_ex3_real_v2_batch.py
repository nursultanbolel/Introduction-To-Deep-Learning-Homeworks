# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 16:19:50 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-4
"""
import getIrısData as getData
import numpy as np
import matplotlib.pyplot as plt
import sigm as S
import nnloss as NL

#now make a real experiment
batchSize=5;
trainingSet, testSet, y1, y2  = getData.getIrisData()

np.save('trainingSet', trainingSet)
np.save('testSet', testSet)
np.save('y1', y1)
np.save('y2', y2)
# train data
input = trainingSet
# train labels
groundTruth = y1
# Learning coefficient
coeff = 0.1
#Number of learning iterations
iterations = 10000

# Calculate weights randomly using seed.
#rand('state',sum(100*clock));

inputLength=input.shape[0]
hiddenN=5; 
outputN=groundTruth.shape[1]
num_layers=3
tol=0.1
# initialize parameters
mystack_1_w=np.random.rand(hiddenN,inputLength)
mystack_1_b=np.random.rand(hiddenN,1)

mystack_2_w=np.random.rand(outputN,hiddenN)
mystack_2_b=np.random.rand(outputN,1)

outputStack1 = np.zeros((5, 1))
outputStack2 = np.zeros((5, 1))
outputStack3 = np.zeros((3, 1))

p=np.zeros((3, 1))

gradStack1_epsilon = np.zeros((5, 1))
gradStack2_epsilon = np.zeros((3, 1))

inputs=np.zeros((5, 1))

plt.Figure()
arr_x=np.zeros((1, 100))
arr_y=np.zeros((1, 100))
counter=0

for i in range(iterations):
    err=0
    for j in range(0,y1.shape[0],batchSize):
        data=input[:,j:j+batchSize]
        labels=groundTruth[j:j+batchSize,:]
        cost=0
        
        for kk in range(batchSize):
            inputs[0:5, 0]=data[:,kk]
            outputStack1 = inputs
              # forward propagation
            outputStack2=np.subtract(np.dot(mystack_1_w,outputStack1),mystack_1_b)
            outputStack2=S.sigm(outputStack2)
            
            outputStack3=np.subtract(np.dot(mystack_2_w,outputStack2),mystack_2_b)
            outputStack3=S.sigm(outputStack3)
            # backward propagation
            p = outputStack3
            epsilon = NL.nnloss(labels[kk:kk+1,:].T, p, 1)
            cost = NL.nnloss(groundTruth[kk:kk+1,:].T, p, 0);
            err = err+cost
            
            if j==0:
                gradStack2_epsilon=np.multiply(np.multiply(outputStack3,(1 - outputStack3)),epsilon)
                epsilon=np.dot(mystack_2_w.T,gradStack2_epsilon)
                
                gradStack1_epsilon=np.multiply(np.multiply(outputStack2,(1 - outputStack2)),epsilon)
                epsilon=np.dot(mystack_1_w.T,gradStack1_epsilon)
                
            else:
                gradStack2_epsilon = gradStack2_epsilon + np.multiply(np.multiply(outputStack3,(1 - outputStack3)),epsilon)
                epsilon=np.dot(mystack_2_w.T,gradStack2_epsilon)
                
                gradStack1_epsilon = gradStack1_epsilon + np.multiply(np.multiply(outputStack2,(1 - outputStack2)),epsilon)
                epsilon=np.dot(mystack_1_w.T,gradStack1_epsilon)
                
            gradStack2_epsilon=gradStack2_epsilon/batchSize
            gradStack1_epsilon=gradStack1_epsilon/batchSize
             
            cost=cost/batchSize
            err = err+cost
            # Update weights
            mystack_1_w = mystack_1_w + np.multiply(coeff, np.dot(gradStack1_epsilon, (outputStack1.T)))
            mystack_1_b = mystack_1_b + np.multiply((coeff*(-1)), gradStack1_epsilon)
            
            mystack_2_w = mystack_2_w + np.multiply(coeff, np.dot(gradStack2_epsilon, (outputStack2.T)))
            mystack_2_b = mystack_2_b + np.multiply((coeff*(-1)), gradStack2_epsilon)

    if i%100 == 0:
        arr_x[0,counter] = i
        arr_y[0,counter] = err
        counter+=1
    
    if abs(err) < tol :
        break
plt.plot(arr_x[0,:],arr_y[0,:], 'b-')
plt.axis([0, 10000, 0, 100])
plt.show()
       
np.save('mystack_1_w', mystack_1_w)
np.save('mystack_2_w', mystack_2_w)
np.save('mystack_1_b', mystack_1_b)
np.save('mystack_2_b', mystack_2_b)

mystack_1_w = np.load('mystack_1_w.npy')
mystack_2_w = np.load('mystack_2_w.npy')
mystack_1_b = np.load('mystack_1_b.npy')
mystack_2_b = np.load('mystack_2_b.npy')

testSet = np.load('testSet.npy')
y2 = np.load('y2.npy')

#test the code
input = testSet
tol=0.1
groundTruth = y2
out=np.zeros((y2.shape[0],y2.shape[1]))
# outputStack = np.random.rand(num_layers,1) 
count=0

for j in range(y2.shape[0]):
    inputs[:, 0] = input[:, j]
    outputStack1 = inputs
    # forward propagation
    outputStack2=np.subtract(np.dot(mystack_1_w,outputStack1),mystack_1_b)
    outputStack2=S.sigm(outputStack2)
        
    outputStack3=np.subtract(np.dot(mystack_2_w,outputStack2),mystack_2_b)
    outputStack3=S.sigm(outputStack3)
    
    out[j,:]=outputStack3.T
    epsilon=np.subtract(groundTruth[j,:],out[j,:])
    err=np.sum(pow(epsilon,2))
    
    if err<tol:
        count=count+1
        
acc = (count/out.shape[0])*100    
print('accuracy of system: ', acc)

