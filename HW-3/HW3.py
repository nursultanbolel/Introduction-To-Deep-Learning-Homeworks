# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:11:16 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-3
"""

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import imageio
import cv2

def RGB2GRAY_RESIZE(A,size1,size2):
#''' This function is used to change an input image. Firstly the input image is turned to gray scale, resized to size1*size2 
#and then converted a vector
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
#''' This function is used to make smaller the  values ​​of A . Some image has high pixel values after translating the RGB2GRAY
#transaction.This functions scaled value of vector between 0 and 1
#    
#        Args:
#        A: is a vector of image
#
#        Returns:
#        The return value is an vector. it's value is between 0 and 1
#'''
     dd = np.max(A)
     return A/dd
    
def ReadImage_Laptop():
#''' This function is used to prepare training data.This function is for laptop images. Label value of laptop is 3.
#Firstly trainig images is reading from a file. Secondly the images is turned to gray scale, resized to 128*128 and then converted a vector
#for training by RGB2GRAY_RESIZE function. The vector assigned to train_images array which saves laptop trainig data in between 0 and
#79 index value.If after calling RGB2GRAY_RESIZE the vector has high value, CHANGE_IMAGE_PIXEL is called.Finally 3 assigned to train_class array
#which is labels of training images.
#'''   
    a=0
    for i in range(1,82):
        if(i!=7):
            if(i<10):
                Orginal_Image= imageio.imread('train/laptop/image_000' + (i).__str__() +'.jpg')
        
            if(i>=10):
                Orginal_Image = imageio.imread('train/laptop/image_00' + (i).__str__() +'.jpg')
            
            #ShowImage(Orginal_Image)
            Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Orginal_Image,128,128)
            
            if(i==3 or i==53):
               Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
            
            train_images[i-1-a]=Flatten_Resized_Image_gray
            train_class[i-1-a]=3
            
            
        else:
            a=1
            
def ReadImage_Chair():
#''' This function is used to prepare training data.This function is for chair images. Label value of chair is 2.
#Firstly trainig images is reading from a file. Secondly the images is turned to gray scale, resized to 128*128 and then converted a vector
#for training by RGB2GRAY_RESIZE function. The vector assigned to train_images array which saves laptop trainig data in between 80 and
#140 index value.If after calling RGB2GRAY_RESIZE the vector has high value, CHANGE_IMAGE_PIXEL is called.Finally 2 assigned to train_class array
#which is labels of training images.
#'''     
    for i in range(2,63):
        if(i<10):
            Orginal_Image= imageio.imread('train/chair/image_000' + (i).__str__() +'.jpg')
        if(i>=10):
            Orginal_Image = imageio.imread('train/chair/image_00' + (i).__str__() +'.jpg')
            
        Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Orginal_Image,128,128)
        
        if(i==28):
            Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
        
        train_images[i+78]=Flatten_Resized_Image_gray
        train_class[i+78]=2
        
def ReadImage_Butterfly():
#''' This function is used to prepare training data.This function is for chair images. Label value of butterfly is 1.
#Firstly trainig images is reading from a file. Secondly the images is turned to gray scale, resized to 128*128 and then converted a vector
#for training by RGB2GRAY_RESIZE function. The vector assigned to train_images array which saves butterfly trainig data in between 141 and
#230 index value.If after calling RGB2GRAY_RESIZE the vector has high value, CHANGE_IMAGE_PIXEL is called.Finally 1 assigned to train_class array
#which is labels of training images.
#'''   
    a=0
    for i in range(1,92):
        if(i!=31):
            if(i<10):
                Orginal_Image= imageio.imread('train/butterfly/image_000' + (i).__str__() +'.jpg')
        
            if(i>=10):
                Orginal_Image = imageio.imread('train/butterfly/image_00' + (i).__str__() +'.jpg')
            
            #ShowImage(Orginal_Image)
            Flatten_Resized_Image_gray=RGB2GRAY_RESIZE(Orginal_Image,128,128)
            
            if(i==9 or i==43 or i==46 or i==59 or i==83):
               Flatten_Resized_Image_gray=CHANGE_IMAGE_PIXEL(Flatten_Resized_Image_gray)
            
            train_images[i+140-a]=Flatten_Resized_Image_gray
            train_class[i+140-a]=1
            
        else:
            a=1
            
def KNN(x_train, y_train, sample_test, k):
#''' This function is used to detect an object, laptop,chair or butterfly using KNN Algorithm. All steps of the algorithm
#are applied in this function. Euclidean distance is used to calculate similarity. 
#
#        Args:
#        x_train: is a matrix which consists of vectors of images
#        y_train: includes class information(1 for butterfly, 2 for chair and 3 for laptop)
#        sample_test: is vector form of test images 
#        k: is the nearest neighbor size
#        
#
#        Returns:
#        The return value is class number(0-1-2)
#'''    
    diff1=np.ones((1,16384))
    diff2=np.ones((1,16384))
    diff1=sample_test
    distance=np.ones((231,1))
    classter=np.zeros((3,1))
    
    for i in range(231):
        diff2=x_train[i]
        sub=np.subtract(diff1,diff2)
        mul=np.multiply(sub,sub)
        deger=np.sum(mul)
        distance[i,0]=np.sqrt(deger)
        
    distance=np.hstack((distance,y_train))
    
    sortedArr = distance[distance[:,0].argsort()]

    for i in range(k):
        if sortedArr[i,1] == 1:
            classter[0,0]=classter[0,0]+1
        if sortedArr[i,1] == 2:
            classter[1,0]=classter[1,0]+1
        if sortedArr[i,1] == 3:
           classter[2,0]=classter[2,0]+1
    
    max_cluster=np.max(classter)
    
    if max_cluster == classter[0,0]:
        return 1
    if max_cluster == classter[1,0]:
        return 2
    if max_cluster == classter[2,0]:
        return 3
    
def Test_Function(path):
#''' This function takes an path of image and then changes the image vector for detection of butterfly, cahir and laptop . Finally shows suitable
#outputs on Console screen in the Spyder.
#
#        Args:
#        path: path of test image
#
#''' 
    Test_Image= imageio.imread(path)
    print('\n--> Test image  is shown below')
    ShowImage(Test_Image)
    final_test_image=RGB2GRAY_RESIZE(Test_Image,128,128)
    final_test_image=CHANGE_IMAGE_PIXEL(final_test_image)

    result=KNN(train_images,train_class,final_test_image,7)
    print('After KNN algotihm class is ',result)
    
            
def ShowImage(A):
#''' This function is used to show an image on Console screen in Spyder.
#        Args:
#        A: is an image which will shown on Concole screen in Spyder.
#'''
    imgplot = plt.imshow(A)
    plt.show()  
    
   
train_images=np.zeros((231,16384))
train_class=np.ones((231,1))


ReadImage_Laptop()
ReadImage_Chair()
ReadImage_Butterfly()

Test_Function('test/image_0001.jpg')
Test_Function('test/image_0007.jpg')
Test_Function('test/image_0031.jpg')
