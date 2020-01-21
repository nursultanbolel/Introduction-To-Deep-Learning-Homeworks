# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 23:24:31 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-2
"""
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import imageio
import cv2


def RGB2GRAY_RESIZE(A,size1,size2):
#''' This function is used to change an input image. Firstly the input image is turned to gray scale, resized to size1*size2 
#and then converted to a vector
#
#        Args:
#        A: is an image
#        size1: is an integer number which is used to obtain height of new image
#        size2: is an integer number which is used to obtain width of new image
#
#        Returns:
#        The return value is a vector of a changed image.
#'''
     A = color.rgb2gray(A)
     A=cv2.resize(A,(size1,size2))
     return A.flatten()
 
def CHANGE_IMAGE_PIXEL(A):
#''' This function is used to make smaller the  sizes ​​of A . Some image has high pixel values after translating the RGB2GRAY
#transaction.This functions scales value of vector between 0 and 1
#    
#        Args:
#        A: is a vector of image
#
#        Returns:
#        The return value is an vector. it's value is between 0 and 1
#'''
     dd = np.max(A)
     return A/dd
     
def ReadImage_Cannon():
#''' This function is used to prepare training data.This function is for cannon images. Label value of cannon is 0.
#Firstly trainig images is reading from a file. Secondly the images is turned to gray scale, resized to 128*128 and then converted a vector
#for training by RGB2GRAY_RESIZE function. The vector assigned to train_images array which saves cannon trainig data in between 0 and
#41 index value.If after calling RGB2GRAY_RESIZE, the vector has high value CHANGE_IMAGE_PIXEL is called.Finally 0 assigned to target array
#which is label of cannon images.
#'''   
    a=0
    for i in range(1,44):
        if(i!=21):
            if(i<10):
                Orginal_Image= imageio.imread('train/cannon/image_000' + (i).__str__() +'.jpg')
            if(i>=10):
                Orginal_Image = imageio.imread('train/cannon/image_00' + (i).__str__() +'.jpg')
               
            Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Orginal_Image,128,128)
           
            if(i==3 or i==4):
               Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
            
            train_images[i-1-a]=Flatten_Resized_Image_gray
            t[i-1-a]=0
            
        else:
            a=1
            
def ReadImage_CellPhone():
#''' This function is used to prepare training data.This function is for cellphone images. Label value of cell phone is 1.
#Firstly trainig images is reading from a file. Secondly the images is turned to gray scale, resized to 128*128 and then converted a vector
#for training by RGB2GRAY_RESIZE function and then the vector assigned to train_images array which saves cellphone trainig data in between 42 and
#99 index value. If after calling RGB2GRAY_RESIZE, the vector has high value CHANGE_IMAGE_PIXEL is called.Finally 1 assigned to target which is
#label of cellphone images.
#'''     
    for i in range(2,60):
        if(i<10):
            Orginal_Image= imageio.imread('train/cellphone/image_000' + (i).__str__() +'.jpg')
        if(i>=10):
            Orginal_Image = imageio.imread('train/cellphone/image_00' + (i).__str__() +'.jpg')
            
        Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Orginal_Image,128,128)
        
        if(i==12):
            Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
        
        train_images[i+40]=Flatten_Resized_Image_gray
        
def Activation_function_sigmod(sum):
#''' This function is an Activation function in NN. Sigmoid function is used for activation function.
#Sigmoid function formula = g(x)=1/(1+e^x) --> x=sum
#    
#        Args:
#        sum: is sum of products of input values and weight values in NN.
#
#        Returns:
#        The return value is output of activation function
#'''
    return 1 / (1 + np.exp(-sum))

def Derivative_Sigmoid(y):
#''' This function is used to find g'(y) is refers to derivative of sigmoid function.
#g'(y)=g(y)*(1-g(y))
#    
#        Args:
#        sum: is output of activation function
#
#        Returns:
#        The return is derivative of output 
#'''
    return Activation_function_sigmod(y)*(1-Activation_function_sigmod(y))

def trainPerceptron(inputs, t, weights, rho, iterNo):
#''' This function is used to train NN to detect an object, cannon or cellphone using Perceptron Algorithm. All steps of the algorithm
#are applied in this function. Activation function is sigmoid function.
#Firstly inputs and weights are multiplied and summed all values and then used sigmoid function.Output of the sigmoid function is output.
# Secondly error and derivative of sigmoid function are calculated and they are used to calculate delta weights(delata_w). delta weights is used  
#for feed-backward and w_init is updated.
# ---> Calculating error,derivative of sigmoid function,delta weights and updating w_init are for feed-backward process.
# --->Multiplication of inputs and weights,summing of multiplication results and using activation function are for feed-forward process.
# The first and second steps above are repeated until iterNo.  
#
#        Args:
#        inputs: is a matrix which consists of vectors of images and bias value(1)
#        t:  is target array which is equal to 0 or 1
#        weights: Initial weights for the linear discriminant function
#        rho: learning rate
#        iterNo: refers to number of iterations in perceptron algorithm
#
#        Returns:
#'''    
#        The return value is updated vector of weights
    print('--> Trainig is started ...')
    w_init=weights
    for i in range(iterNo):
        delta_w=0;
        h,w=inputs.shape
        for j in range(h):
# --->Multiplication of inputs and weights,summing of multiplication results and using activation function are for feed-forward process.
            sum= np.dot(w_init,np.transpose(inputs[j]))
            output=Activation_function_sigmod(sum)
# ---> Calculating error,derivative of sigmoid function,delta weights and updating w_init are for feed-backward process.
            derivative_output=Derivative_Sigmoid(sum)
            error=t[j,0]-output
            delta_w=rho*error*derivative_output*inputs[j]
            w_init=w_init+delta_w
    print('--> Trainig is ended.')        
    return w_init

def testPerceptron(sample_test, weights):
#''' This function is used to test NN to detect an object cannon or cellphone from a vector of image using Perceptron Algorithm. 
#Activation function is sigmoid function.
#Inputs(sample_test) and weights are multiplied and summed all values and then used sigmoid function.Output of sigmoid function is output.
#
#        Args:
#        sample_test: is a vectors of image and bias value(1)
#        weights: Initial weights for the linear discriminant function. 
#
#        Returns:
#        The return value is output of NN
#''' 
    sum=np.dot(weights,np.transpose(sample_test))
    output = Activation_function_sigmod(sum)
    return  output

def Trashold(result):
#''' This function is used after detection. If output of NN is bigger than 0.5 retuned 1 else returned 0.
#
#        Args:
#        sonuc: is a vectors of images and bias value(1) 
#
#        Returns:
#        The return value is 1 or zero
#''' 
    if(result>0.5):
        return 1
    else :
        return 0
    
def Test_Function(path):
#''' This function takes an path of image and then changes the image vector for detection of cellphone and cannon. Finally show suitable
#outputs on Console screen in Spyder.
#
#        Args:
#        path: path of image
#
#''' 
    Test_Image= imageio.imread(path)
    print('\n--> Test image  is shown below')
    ShowImage(Test_Image)
    Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Test_Image,128,128)
    Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
    final_test_image=np.hstack((Flatten_Resized_Image_gray,1))
    result=testPerceptron(final_test_image,weights)
    print('Output of NN: ',result)
    result=Trashold(testPerceptron(final_test_image,weights))
    print('After Trashold Output of NN: ',result)

def ShowImage(A):
#''' This function is used to show an image on Concole screen in Spyder.
#        Args:
#        A: is an image which will shown on Concole screen in Spyder.
#'''
    imgplot = plt.imshow(A)
    plt.show()    


       
train_images=np.zeros((100,16384))
final_train_images=np.zeros((100,16385))
bias=np.ones((100,1))
t=np.ones((100,1))
weights=np.zeros((1,16385))
weights[:]=0.0001

#Prepearing data for training
ReadImage_Cannon()
ReadImage_CellPhone()
final_train_images=np.hstack((train_images,bias))

#Shuffling training data
final_train_images, t = shuffle(final_train_images, t)

#Training
weights=trainPerceptron(final_train_images, t, weights, 0.001, 1000)

#Testing
Test_Function('test/cannon/image_0021.jpg')
Test_Function('test/cellphone/image_0001.jpg')