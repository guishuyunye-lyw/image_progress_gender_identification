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

# 导入原始图片
def get_img(path):
    images=[]
    for file in glob.glob(path):
        img=plt.imread(file)
        #img=cv2.equalizeHist(img)
        img=autonorm1(img)
        L=img.size
        img.shape=[1,L]
        images.append(img[0])
    return np.array(images)    
    
    
    
# 导入图像的LBP图   
def face_picture_LBP_import(s_path):
    count=0 #图像的数量
    faces_LBPs=[]
    for bmpfile in glob.glob(s_path):
        img=plt.imread(bmpfile)
        img_LBP=LBP(img,2,10)
        count+=1
        faces_LBPs.append(img_LBP)
    faces_LBPs=np.array(faces_LBPs)
    return faces_LBPs,count
    
    
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
        
#直方图拼接
def hist_join(faces_LBP):     #faces_LBP is a 3D matrix    
    hist=[]
    for i in range(faces_LBP.shape[0]):
        img_hist=np.array([])
        img=faces_LBP[i,:,:]
        img_hv44_hist=HVdivide(img,PP=10)
        img_hv22_hist=HVdivide(img,[0,49,99],[0,49,99],PP=10)
        img_hist=np.hstack((img_hv44_hist,img_hv22_hist,Hist_quanju(img,PP=10)))    
        hist.append(img_hist)
    hist=np.array(hist)
    return hist



def hist_pca(faces_hist,a=0.9):
    #直方图拼接的pca
    mean,egiventhist=cv2.PCAComputeVar(faces_hist,a)
    egiventhist=egiventhist.T 
    global HISTPCA_EGI
    HISTPCA_EGI=egiventhist
    faces_hist_pca=[]
    for i in range(faces_hist.shape[0]):
        img=faces_hist[i,:]
        img_hist_pca=np.dot(img,egiventhist)
        img_hist_pca_norm=autonorm1(img_hist_pca)
        faces_hist_pca.append(img_hist_pca_norm)
    faces_hist_pca=np.array(faces_hist_pca)    
    return faces_hist_pca

# 原图像的pca 归一化
def orginal_image_pca_norm(faces,a):  
    mean,egivent=cv2.PCAComputeVar(faces,a)
    egivent=egivent.T
    global ORGPCA_EGI
    ORGPCA_EGI=egivent
    faces_pca_norm=[]
    for i in range(faces.shape[0]):
        img=faces[i,:]
        img_pca=np.dot(img,egivent)
        img_pca_norm=autonorm1(img_pca)
        faces_pca_norm.append(img_pca_norm)
    faces_pca_norm=np.array(faces_pca_norm)
    return faces_pca_norm    



def test_org_pca(grilboy_faces):
    n=grilboy_faces.shape[0]
    pca=[]
    for i in range(n):
        img=grilboy_faces[i,:]
        img_pca=np.dot(img,ORGPCA_EGI)
        img_pca_norm=autonorm1(img_pca)
        pca.append(img_pca_norm)
    pca=np.array(pca)
    return pca

def test_hist_pca(grilboy_hist):
    n=grilboy_hist.shape[0]
    pca_h=[]
    for i in range(n):
        img=grilboy_hist[i,:]
        img_hist_pca=np.dot(img,HISTPCA_EGI)
        img_hist_pca_norm=autonorm1(img_hist_pca)
        pca_h.append(img_hist_pca_norm)
    pca_h=np.array(pca_h)    
    return pca_h
    
## 补充图片
boy_net=get_img('C:\\Users\\WV\\Desktop\\netfaces\\boy\\*.png')
gril_net=get_img('C:\\Users\\WV\\Desktop\\netfaces\\gril\\*.png')
# 
gril_LBP_net,gril_count=face_picture_LBP_import('C:\\Users\\WV\\Desktop\\netfaces\\gril\\*.png')#导入女性图像
boy_LBP_net,boy_count=face_picture_LBP_import('C:\\Users\\WV\\Desktop\\netfaces\\boy\\*.png')#导入男性图像

    
    
faces_new=np.vstack((gril_net,faces,boy_net))
faces_LBP_new=np.vstack((gril_LBP_net,faces_LBP,boy_LBP_net))    
    
    
    
    
    
#////////////////////////////////////////////////////////////////////////////////////////////////////////// 
# 导入测试的原始图片    
boy=get_img('C:\\Users\\WV\\Desktop\\FileRecv2\\boy\\*.png')
gril=get_img('C:\\Users\\WV\\Desktop\\FileRecv2\\gril\\*.png')
grilboy_faces= np.row_stack((gril,boy))    
    
    
# 导入测试图片的LBP图
gril_LBP,gril_count=face_picture_LBP_import('C:\\Users\\WV\\Desktop\\FileRecv2\\gril\\*.png')#导入女性图像
boy_LBP,boy_count=face_picture_LBP_import('C:\\Users\\WV\\Desktop\\FileRecv2\\boy\\*.png')#导入男性图像
grilboy_LBP=np.vstack((gril_LBP,boy_LBP))


# 测试图的hist
grilboy_hist=hist_join(grilboy_LBP)

#///////////////////////////// 原图做测试   ///////////////////////////////////////////////////////////////////
f_hist_pca_90=hist_pca(np.vstack((faces_hist[:435,:],faces_hist[445:1030,:])),0.9)    
f_pca_90=orginal_image_pca_norm(np.vstack((faces[:435,:],faces[445:1030,:])),0.9)  
ff=np.hstack((f_hist_pca_90,f_pca_90))
ytrain=np.array([1]*435+[0]*585)

t_faces_pca_90= test_org_pca(np.vstack((faces[435:445,:],faces[1030:,:])))  
t_hist_pca_90=test_hist_pca(np.vstack((faces_hist[435:445,:],faces_hist[1030:,:])))    
tt=np.hstack((t_hist_pca_90,t_faces_pca_90))


svc = svm.LinearSVC()
svc.fit(ff,ytrain)
p=svc.predict(tt)

#///////////////////////////// 自定义图做测试  ///////////////////////////////////////////////////////////////
faces_hist=hist_join(faces_LBP_new)
faces_hist_pca_90=hist_pca(faces_hist,0.9)    
faces_pca_90=orginal_image_pca_norm(faces_new,0.9)  
faces_hist_pca_90_faces_pca_90=np.hstack((faces_hist_pca_90,faces_pca_90))
target_new=target=np.array([1]*675+[0]*777)


grilboy_faces_pca_90= test_org_pca(grilboy_faces)  
grilboy_hist_pca_90=test_hist_pca(grilboy_hist)    
grilboy_hist_pca_90_grilboy_faces_pca_90=np.hstack((grilboy_hist_pca_90,grilboy_faces_pca_90))

xtrain=faces_hist_pca_90_faces_pca_90
xtest=grilboy_hist_pca_90_grilboy_faces_pca_90

target=np.array([1]*445+[0]*595)



'SVM'
svc = svm.LinearSVC()
svc.fit(xtrain,target_new)
p=svc.predict(xtest)
print p


'randomforest'
clf = RandomForestClassifier(random_state=1,n_estimators=100)
clf = clf.fit(xtrain,target)
p=clf.predict(xtest)
print p




'adaboost'
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(xtrain,target)
p=ada.predict(xtest)
print p

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#SVM分类
    
svc = svm.LinearSVC()
svc.fit(ff,ytrain)
p=svc.predict(tt)

svc.fit(faces_pca_90,target)
tt=svc.predict(grilboy_faces_pca_90)
 
    
    




        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    