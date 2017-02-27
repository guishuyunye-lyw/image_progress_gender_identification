# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:11:06 2015

@author: WV
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
import cv2
import math

def binary2dec(lst):
    L=len(lst)
    dec=0
    lst1=lst[:]
    lst1.reverse()
    for i in range(L):
        dec+=lst1[i]*(2**i)
    return dec
        
        
def min_value_of_binary2dec(lst):
    L=len(lst)
    min_value=binary2dec(lst)
    for i in range(1,L):
        temp_lst=lst[i:]+lst[:i]
        temp_value=binary2dec(temp_lst)
        min_value=min(min_value,temp_value)
    return min_value
        
def LBP_8neigh(face_mat):
    height,width=face_mat.shape
    face_LBP=np.zeros([height-2,width-2],dtype=np.uint8)
    hist=np.zeros([1,256])[0]
    for i in range(1,height-1):
        for j in range(1,width-1):
            neigh=[0,0,0,0,0,0,0,0]           
            neigh[7]=face_mat[i-1][j-1]
            neigh[6]=face_mat[i-1][j]
            neigh[5]=face_mat[i-1][j+1]
            neigh[4]=face_mat[i][j-1]
            neigh[3]=face_mat[i][j+1]
            neigh[2]=face_mat[i+1][j-1]
            neigh[1]=face_mat[i+1][j+1]
            neigh[0]=face_mat[i+1][j+1]
            center=face_mat[i][j]
            lst=[int(item>=center) for item in neigh]
            temp_value=min_value_of_binary2dec(lst)
            face_LBP[i-1,j-1]=int(temp_value)
            hist[temp_value]+=1
    return face_LBP,hist


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
    hist = cv2.calcHist([face_LBP],[0],None,[2**PP],[0.0,float(2**PP-1)]) #直方图柱的范围 
    a,b=hist.shape
    hist.shape=[b,a]
    hist=hist[0]
    return face_LBP,hist
    
                
            
               
#img=Image.open('C:\Users\WV\Desktop\galleryset_f\FY_000121_IEU+00_PM+00_EN_A0_D0_T0_BB_M0_R1_S0.bmp')

Im=plt.imread('L:\\weishi\\face_recognization\\pythoncode\\galleryset_f\\FY_000121_IEU+00_PM+00_EN_A0_D0_T0_BB_M0_R1_S0.bmp')
img_mat=cv2.equalizeHist(Im)    


plt.imshow(img_mat,cmap=cm.gray)

plt.imshow(img_mat[:35,:],cmap=cm.gray)

plt.imshow(img_mat[35:70,:],cmap=cm.gray)

plt.imshow(img_mat[70:,:],cmap=cm.gray)

img_LBP,img_hist=LBP(img_mat,2,12)

plt.imshow(img_LBP,cmap=cm.gray) 



hist = cv2.calcHist([img_LBP],[0],None,[256],[0.0,255.0]) #直方图柱的范围    
cv2.imshow("hist", hist)  
cv2.waitKey(0)  
cv2.destroyAllWindows()      



#####################################################################################################################
def svmclassify(x_train,y_train,x_test,y_test):
    svc = svm.SVC(kernel='rbf',C=4,gamma=0.00125)
    #svc = svm.SVC(kernel='poly',degree=3)
    svc.fit(x_train,y_train)
    svc.predict(x_test)
    return svc.score(x_test,y_test)

# 矩阵归一化处理
def autonorm1(mat):
    M = mat.max()
    m = mat.min()
    norm_mat = (mat-m)*1.0/(M-m)
    return norm_mat

def autonorm2(mat):
    mean = mat.mean()
    var = mat.var()
    norm_mat = (mat-mean)/var
    return norm_mat

# 对每一张图像矩阵做归一化，加标签，female=1，male=0
def face_hist_label(face_array,label):
    face_array_norm = autonorm1(face_array)
    face_LBP,hist=LBP(face_array_norm)
    face_LBP=autonorm1(face_LBP)
    L=face_LBP.size
    face_LBP.shape=[1,L] 
    LBP_list= list(face_LBP[0])
    hist=autonorm1(hist)
    face_hist_list = list(hist)
    LBP_label=LBP_list+[label]
    face_hist_label = face_hist_list+[label]# label 0 stands for male
    face_hist_norm_label = np.array(face_hist_label)
    LBP_label=np.array(LBP_label)
    return face_hist_norm_label,LBP_label

# 导入图像
def face_picture_import(s_path,label):
    female_count=0 #女性图像的数量
    female_faces_norm_label=[]
    LBPs=[]
    for bmpfile in glob.glob(s_path):
        Im=plt.imread(bmpfile)
        img=cv2.equalizeHist(Im)  #直方图均衡化
        img_hist_label,LBP=face_hist_label(img,label)
        img_hist_list=list(img_hist_label)
        LBP=list(LBP)
        female_count+=1
        female_faces_norm_label.append(img_hist_list)
        LBPs.append(LBP)
    female_faces_norm_label_array=np.array(female_faces_norm_label)
    LBPs=np.array(LBPs)
    return female_faces_norm_label_array,LBPs,female_count
    
female_hist,female_LBP,female_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_f\\*.bmp',1)#导入女性图像

male_hist,male_LBP,male_count=face_picture_import('L:\\weishi\\face_recognization\\pythoncode\\galleryset_m\\*.bmp',0)#导入男性图像

faces_label_array = np.row_stack((male_hist,female_hist)) # 男女图像拉成一行 组合在一起
faces_LBP= np.row_stack((male_LBP,female_LBP))



  
NN = female_count+male_count
X=faces_label_array[:,:-1]
Y=faces_label_array[:,-1]
num_fold=10



NN = female_count+male_count
XX=faces_LBP[:,:-1]
YY=faces_LBP[:,-1]
num_fold=10


def m():
    folds = KFold(NN,n_folds=num_fold,shuffle=True)
    svm_score=0
    for train_index, test_index in folds:
        X_train = XX[train_index]
        X_test  = XX[test_index]
        Y_train = YY[train_index]
        Y_test = YY[test_index]
        svm_single=svmclassify(X_train,Y_train,X_test,Y_test)
        svm_score +=svm_single
        print 'svm_single',svm_single
    return svm_score/num_fold



svms=0
n=10
for i in range(n):
    s=m()
    print 's=',s
    svms+=s
print "score of svm =",svms/n


hh=np.array([[1,2,3],[4,5,6],[7,8,9]])

folds = KFold(50,n_folds=10,shuffle=True)
for train_index, test_index in folds: 
    print train_index,test_index

    
hist = cv2.calcHist([hh],[0],None,[256],[0.0,255.0]) #直方图柱的范围    
    
        
        
        