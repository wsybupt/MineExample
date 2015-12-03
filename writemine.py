#!/usr/bin/env python
import numpy as np
import h5py
 
a=np.loadtxt('data.txt')


tmpA=np.zeros((1599,8))
tmpA[:,0:4]=a[:,0:4]
tmpA[:,4:8]=a[:,5:9]
normA=np.zeros(np.shape(tmpA))

minVals=tmpA.min(0)
maxVals=tmpA.max(0)
ranges=maxVals-minVals
m=tmpA.shape[0]
normA=tmpA-np.tile(minVals,(m,1))
normA=normA/np.tile(ranges,(m,1))

normA=normA.reshape(1599,1,8,1)
a=a.reshape(1599,1,9,1)
b=np.loadtxt('label.txt')
b[:,0]=b[:,0]*500
b[:,1]=b[:,1]*10
b=b.reshape(1599,2)


#f=h5py.File("train.h5","r")
#x=f["data"]
f2=h5py.File("mine.h5","w")
f2.create_dataset("data",data=normA,dtype='float32')
#f2.create_dataset("data",data=a,dtype='float32')
f2.create_dataset("label",data=b,dtype='float32')

trainData=normA[0:1000]
trainLabel=b[0:1000]
testData=normA[1000:]
testLabel=b[1000:]

f3=h5py.File("train.h5","w")
f3.create_dataset("data",data=trainData,dtype='float32')
f3.create_dataset("label",data=trainLabel,dtype='float32')

f4=h5py.File("test.h5","w")
f4.create_dataset("data",data=testData,dtype='float32')
f4.create_dataset("label",data=testLabel,dtype='float32')


