import pandas as pd
import numpy as np
from classifier import classifier
from svmMLiA import smoPK, calcWs

class svm_basic(classifier):
    def __init__(self, C, toler, maxIter):
        self.maxIter = maxIter
        self.C = C
        self.toler = toler
        self.alphas = None
        self.weights = None
        self.b = 0
        
    def fit(self, dataArr, LabelArr):
        self.b, self.alphas = smoPK(dataArr,LabelArr,self.C, self.toler, self.maxIter)
        self.weights = calcWs(self.alphas, dataArr,LabelArr)
        
    def predict(self, dataArr, labelArr): #dataArr = test_x, labelArr = train_y
        datMat=np.mat(dataArr); 
        labelMat = np.mat(labelArr).transpose() #Y
        svInd=np.nonzero(self.alphas.A>0)[0]
        print(svInd)
        sVs=datMat[svInd] #get matrix of only support vectors
        labelSV = labelMat[svInd];
        print("there are %d Support Vectors" % sVs.shape[0])
        m,n = datMat.shape
        for i in range(m):
            NK = sVs*datMat[i,:].T #non-kernel version
            predict = NK.T * np.multiply(labelSV,self.alphas[svInd]) + self.b