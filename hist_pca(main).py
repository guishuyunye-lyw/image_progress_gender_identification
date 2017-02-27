# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 22:09:00 2015

@author: lifeng
"""

import numpy as np
import random
from sklearn import datasets,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from PIL import Image
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2


############# 二进制转化为十进制
def binary2dec(lst):
    L=len(lst)
    dec=0
    lst1=lst[:]
    lst1.reverse()
    for i in range(L):
        dec+=lst1[i]*(2**i)
    return dec
    
#  旋转不变性 二进制转化为十进制
def min_value_of_binary2dec(lst):
    L=len(lst)
    min_value=binary2dec(lst)
    for i in range(1,L):
        temp_lst=lst[i:]+lst[:i]
        temp_value=binary2dec(temp_lst)
        min_value=min(min_value,temp_value)
    return min_value
    
# 圆形 LBP 特征   
def LBP(face_mat,R=2,PP=8):
    '''
    R is the radius，PP is the number of points
    '''
    height,width=face_mat.shape
    pi=math.pi
    face_LBP=np.zeros([height,width],dtype=np.uint8)
    for x in range(height):
        for y in range(width):
            center=face_mat[x,y]
            er=list()
            for p in range(1,PP+1):     
                p=float(p)
                xp= x+R*math.cos(2*pi*(p/PP))
                yp= y-R*R*math.sin(2*pi*(p/PP))
                xp_low=min(max(math.floor(xp),0),height-1)
                xp_upper=min(max(math.ceil(xp),0),height-1)
                yp_low=min(max(math.floor(yp),0),width-1)
                yp_upper=min(max(math.ceil(yp),0),width-1)
                dx=xp-xp_low
                dy=yp-yp_low
                f00=face_mat[xp_low,yp_low]
                f01=face_mat[xp_low,yp_upper]
                f11=face_mat[xp_upper,yp_upper]
                f10=face_mat[xp_upper,yp_low]
                pixel=f00*(1-dx)*(1-dy)+f01*(1-dx)*dy+f11*dx*dy+f10*dx*(1-dy)
                if pixel>=center:
                    er+=[1]
                else:
                    er+=[0]
                face_LBP[x,y]=min_value_of_binary2dec(er)
    return face_LBP


# 归一化
def autonorm1(mat):
    M = mat.max()
    m = mat.min()
    norm_mat = (mat-m)*1.0/(M-m)
    return norm_mat
    
# 导入图像    
def face_picture_import(s_path):
    count=0 #图像的数量
    faces_LBPs=[]
    for bmpfile in glob.glob(s_path):
        Im=plt.imread(bmpfile)
        img=cv2.equalizeHist(Im)  #直方图均衡化
        img_LBP=LBP(img,2,16)
        count+=1
        faces_LBPs.append(img_LBP)
    faces_LBPs=np.array(faces_LBPs)
    return faces_LBPs,count
    
female_LBP,female_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_f\\*.bmp')#导入女性图像
male_LBP,male_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_m\\*.bmp')#导入男性图像
    
faces_LBP=np.vstack((female_LBP,male_LBP))
    
#均衡模式
def UP(lst):
    L=len(lst)
    lst.append(lst[0])
    count=0
    for i in range(L):
        if lst[i]!=lst[i+1]:
            count+=1
    return count
        
def UniformPattern(PP):
    u=[]
    for num in range(2**PP):
        c=bin(num)[2:]
        lst=[int(i) for i in c]
        L=len(c)
        k=PP-L
        lst.reverse()
        while k:
            lst.append(0)
            k-=1
        lst.reverse()
        if UP(lst)<=2:
            u.append(1)
        else:
            u.append(0)
    return u
    
    

#均衡模式直方图
def UPhist(img,PP=8):
    img_hist = cv2.calcHist([img],[0],None,[2**PP],[0.0,2**PP-1])
    l,w=img_hist.shape
    img_hist.shape=[w,l]
    img_hist=img_hist[0]
    u=UniformPattern(PP)
    u_t=[i for i in range(len(u)) if u[i]]
    u_f=[i for i in range(len(u)) if 1-u[i]]
    uphist=[]
    for i in range(len(u_t)):
        uphist.append(img_hist[u_t[i]])
    uphist.append(sum(img_hist[u_f]))
    return np.array(uphist)
    
    

#垂直切分
def Vdivide(img,a=35,b=70,PP=8):
    upper=img[a,:]
    middle=img[a:b,:]
    low=img[b:,:]
    upper_hist = cv2.calcHist([upper],[0],None,[2**PP],[0.0,2**PP-1]) 
    middle_hist = cv2.calcHist([middle],[0],None,[2**PP],[0.0,2**PP-1]) 
    low_hist = cv2.calcHist([low],[0],None,[2**PP],[0.0,2**PP-1]) 
    l,w=low_hist.shape
    upper_hist.shape=[w,l]
    middle_hist.shape=[w,l]
    low_hist.shape=[w,l]
    upper_hist=upper_hist[0]
    middle_hist=middle_hist[0]
    low_hist=low_hist[0]
    img_hist=np.hstack((upper_hist,middle_hist,low_hist))
    return img_hist
#水平垂直切分  
def HVdivide(img,vscale=[0,24,49,74,99],hscale=[0,24,49,74,99],PP=8):
    img_hist=np.array([])
    for h in range(1,len(hscale)):
        for v in range(1,len(vscale)):
            img_hv=img[hscale[h-1]:hscale[h],vscale[v-1]:vscale[v]]
            img_hv_hist = UPhist(img_hv,PP)
            img_hv_hist=autonorm1(img_hv_hist)
            img_hist=np.hstack((img_hist,img_hv_hist))
    return img_hist
#全局的直方图
def Hist_quanju(img,PP=8):
    img_hist = UPhist(img,PP)
    img_hist=autonorm1(img_hist)
    return img_hist
    
    
count=faces_LBP.shape[0]    
    
#直方图拼接
faces_hist=[]
for i in range(count):
    img_hist=np.array([])
    img=faces_LBP[i,:,:]
    img_hv44_hist=HVdivide(img,PP=10)
    img_hv22_hist=HVdivide(img,[0,49,99],[0,49,99],PP=10)
    img_hist=np.hstack((img_hv44_hist,img_hv22_hist,Hist_quanju(img,PP=10)))    
    faces_hist.append(img_hist)
faces_hist=np.array(faces_hist)

def hist_pca(faces_hist,a=0.9):
    #直方图拼接的pca
    mean,egiventhist=cv2.PCAComputeVar(faces_hist,a)
    egiventhist=egiventhist.T    
    faces_hist_pca=[]
    for i in range(count):
        img=faces_hist[i,:]
        img_hist_pca=np.dot(img,egiventhist)
        img_hist_pca_norm=autonorm1(img_hist_pca)
        faces_hist_pca.append(img_hist_pca_norm)
    faces_hist_pca=np.array(faces_hist_pca)    
    return faces_hist_pca
    

 
# 原图像的pca
def orginal_image_pca(a):  
    mean,egivent=cv2.PCAComputeVar(faces,a)
    egivent=egivent.T
    faces_pca=[]
    for i in range(1040):
        img=faces[i,:]
        img_pca=np.dot(img,egivent)
        faces_pca.append(img_pca)
    faces_pca=np.array(faces_pca)
    return faces_pca

# 原图像的pca 归一化
def orginal_image_pca_norm(a):  
    mean,egivent=cv2.PCAComputeVar(faces,a)
    egivent=egivent.T
    faces_pca_norm=[]
    for i in range(1040):
        img=faces[i,:]
        img_pca=np.dot(img,egivent)
        img_pca_norm=autonorm1(img_pca)
        faces_pca_norm.append(img_pca_norm)
    faces_pca_norm=np.array(faces_pca_norm)
    return faces_pca_norm    
    
    
#SVM分类

target=np.array([1]*445+[0]*595)
    
def svmclassify_linear(x_train,y_train,x_test,y_test):
    svc = svm.LinearSVC()
    svc.fit(x_train,y_train)
    return svc.score(x_test,y_test)

def randomforestclassify(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier(random_state=1,n_estimators=50)
    clf = clf.fit(x_train,y_train)
    return clf.score(x_test,y_test)

def svmclassify_rbf(x_train,y_train,x_test,y_test):
    svc = svm.SVC(kernel='rbf',C=4,gamma=0.0015)
    svc.fit(x_train,y_train)
    return svc.score(x_test,y_test)

num_fold=10
def m_linear(x,y):
    folds = KFold(x.shape[0],n_folds=num_fold,shuffle=True)
    rf_score=0
    for train_index, test_index in folds:
        X_train = x[train_index]
        X_test  = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        rf_single=svmclassify_linear(X_train,Y_train,X_test,Y_test)
        rf_score +=rf_single
        print 'rf_single',rf_single
        #break
    return rf_score/num_fold


def m_rbf(x,y):
    folds = KFold(count,n_folds=num_fold,shuffle=True)
    rf_score=0
    for train_index, test_index in folds:
        X_train = x[train_index]
        X_test  = x[test_index]
        Y_train = y[train_index]
        Y_test = y[test_index]
        rf_single=svmclassify_rbf(X_train,Y_train,X_test,Y_test)
        rf_score +=rf_single
        print 'rf_single',rf_single
        #break
    return rf_score/num_fold


        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    