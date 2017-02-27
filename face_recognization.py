# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
from sklearn import datasets,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from PIL import Image

def svmclassify(x_train,y_train,x_test,y_test):
    svc = svm.SVC(kernel='linear')
    #svc = svm.SVC(kernel='poly',degree=3)
    svc.fit(x_train,y_train)
    svc.predict(x_test)
    return svc.score(x_test,y_test)


def randomforestclassify(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier(random_state=1,n_estimators=100)
    clf = clf.fit(x_train,y_train)
    return clf.score(x_test,y_test)
''' 
digits = datasets.load_digits()

N = digits.data.shape[0]
p = 2/3.0
n =int( N*p)
x_train,y_train = digits.data[:n], digits.target[:n]
x_test,y_test = digits.data[n:], digits.target[n:]

svm_score = svmclassify(x_train,y_train,x_test,y_test)
print ('score for svm=',svm_score)
RF_score = randomforestclassify(x_train,y_train,x_test,y_test)
print ('score for randomforest=',RF_score)


########################################### the above code is the test dataset of digits ########################################################
'''


num_female = 205
num_male = 196
femalefaces = []
malefaces = []
for i in range(num_female):
    face = Image.open("F:\\faces\\histfemale=0=0\\new{}.png".format(i))
    femalefaces.append(face)
    
for j in range(num_male):
    face = Image.open("F:\\faces\\histmale=0=1\\new{}.png".format(j))
    malefaces.append(face)

female = femalefaces[0]
male = malefaces[0]
ff = np.array(female)
mm = np.array(male)
Length = ff.shape[0]*ff.shape[1]

def autonorm1(mat):
    M = mat.max()
    m = mat.min()
    norm_mat = (mat-m)/(M-m)
    return norm_mat

def autonorm2(mat):
    mean = mat.mean()
    var = mat.var()
    norm_mat = (mat-mean)/var
    return norm_mat


def face_add_label(faces,label):
    faces_array = []
    if label==0:
        num=num_male
    else:
        num=num_female    
    for i in range(num):
        face = np.array(faces[i])
        face_array_norm = autonorm1(face)
        face_array_norm.shape=(1,Length)
        face_array = list(face_array_norm[0])
        face_array_label = face_array+[label]# label 0 stands for male
        faces_array.append(face_array_label)
        facesarray = np.array(faces_array)
    return facesarray


malefaces_array = face_add_label(malefaces,0)
femalefaces_array = face_add_label(femalefaces,1)
faces_label_array = np.row_stack((malefaces_array,femalefaces_array))
random.shuffle(faces_label_array)

NN = num_female+num_male
X=faces_label_array[:,:-1]
Y=faces_label_array[:,-1]
num_fold=10
def main():
    folds = KFold(NN,n_folds=num_fold,shuffle=True)
    svm_score=0
    RF_score=0
    for train_index, test_index in folds:
        X_train = X[train_index]
        X_test  = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        svm_score += svmclassify(X_train,Y_train,X_test,Y_test)
        RF_score += randomforestclassify(X_train,Y_train,X_test,Y_test)
    return svm_score/num_fold,RF_score/num_fold
    print(svm_score/num_fold,RF_score/num_fold)
  
svms=0
rfs=0  
for i in range(10):
    s,r=main()
    svms+=s
    rfs+=r
print("score of svm=",svms/10)
print("score of randomforest=",rfs/10)    
    

#print ('score for svm=',svm_score/num_fold)
#print ('score for randomforest=',RF_score/num_fold)



'''
pp = 0.9
nn =int(NN*pp)
X_train,Y_train = faces_label_array[:nn,:-1], faces_label_array[:nn,-1]
X_test,Y_test = faces_label_array[nn:,:-1], faces_label_array[nn:,-1]


svm_score = svmclassify(X_train,Y_train,X_test,Y_test)
print ('score for svm=',svm_score)

RF_score = randomforestclassify(X_train,Y_train,X_test,Y_test)
print ('score for randomforest=',RF_score)
'''























