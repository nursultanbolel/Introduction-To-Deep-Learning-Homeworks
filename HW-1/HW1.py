# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:54:09 2019

@Name-Surname: NUR SULTAN BOLEL
@Homework: Homework-1
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import imageio

def ShowImage(A):
    imgplot = plt.imshow(A)
    plt.show()


def FlattenAndAfter(A):
    A[np.isnan(A)] = 0
    A[np.isinf(A)] = 0
    
    dd = np.max(A.flatten() )
    A = A/dd
    A = A*255
    A = np.uint8(A)


def imageFlipVertically(A):
    A = np.float32(A)
    h,w=A.shape
    F_V_Image=A.copy()
    
    for i in range(h):
        for j in range(w):
            F_V_Image[i,j]=A[h-1-i,j]
            
    return F_V_Image


def imageFlipHorizontally(A):
    A = np.float32(A)
    h,w=A.shape
    F_H_Image=A.copy()
    
    for i in range(h):
        for j in range(w):
            F_H_Image[i,j]=A[i,w-1-j]
            
    return F_H_Image


def imageRotateLeft90(A):
    A = np.float32(A)
    h,w=A.shape
    RL_90_Image=A.copy()
    RL_90_Image=RL_90_Image.reshape(w, h)
    
    for i in range(h):
        for j in range(w):
            RL_90_Image[w-1-j,i]=A[i,j]
            
    return RL_90_Image


def imageRotateRight90(A):
    A = np.float32(A)
    h,w=A.shape
    RR_90_Image=A.copy()
    RR_90_Image=RR_90_Image.reshape(w, h)
    
    for i in range(h):
        for j in range(w):
            RR_90_Image[j,h-1-i]=A[i,j]
    
    return RR_90_Image


def imageHalf(A):
    A = np.float32(A)
    h,w=A.shape
    h=int(h/2)
    w=int(w/2)
    Half_Image = np.arange(h*w).reshape(h, w)
    Half_Image = Half_Image.astype('float32') 

    for i in range(h):
        for j in range(w):
            Half_Image[i,j]=(A[2*i,2*j]+A[(2*i)+1,2*j]+A[2*i,(2*j)+1]+A[(2*i)+1,(2*j)+1])/4
            
    return Half_Image


'''
SHOWS AND TURNING TO GRAY ORGINAL IMAGE
'''
Orginal_Image = imageio.imread('hw1.png')
Orginal_Image_gray = color.rgb2gray(Orginal_Image)

print ('#####################\n# ORGINAL CAT IMAGE #\n#####################')
ShowImage(Orginal_Image)

'''
FLIPS IMAGE VERTICALLY 
'''
print ('################################\n# FLIPPED CAT IMAGE VERTICALLY #\n################################')
       
F_V_Image=imageFlipVertically(Orginal_Image_gray)
FlattenAndAfter(F_V_Image)
ShowImage(F_V_Image)

'''
FLIPS IMAGE HORIZONTALLY 
'''
print ('##################################\n# FLIPPED CAT IMAGE HORIZONTALLY #\n##################################')
       
F_H_Image=imageFlipHorizontally(Orginal_Image_gray)
FlattenAndAfter(F_H_Image)
ShowImage(F_H_Image)

'''
ROTATES IMAGE TO LEFT BY 90 DEGREE 
'''
print ('##########################################\n# ROTATED CAT IMAGE TO LEFT BY 90 DEGREE #\n##########################################')
RL_90_Image=imageRotateLeft90(Orginal_Image_gray)
FlattenAndAfter(RL_90_Image)
ShowImage(RL_90_Image)

'''
ROTATES IMAGE TO RIGHT BY 90 DEGREE 
'''
print ('###########################################\n# ROTATED CAT IMAGE TO RIGHT BY 90 DEGREE #\n###########################################')
RR_90_Image=imageRotateRight90(Orginal_Image_gray)
FlattenAndAfter(RR_90_Image)
ShowImage(RR_90_Image)

'''
RESIZES INPUT IMAGE TO HALF BY KEEPING ASPECT RATIO
'''
print ('#######################################################\n# RESIZED INPUT IMAGE TO HALF BY KEEPING ASPECT RATIO #\n#######################################################')
Half_Image=imageHalf(Orginal_Image_gray)
FlattenAndAfter(Half_Image)
ShowImage(Half_Image)
