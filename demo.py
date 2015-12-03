#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py

caffe_root = '/usr/share/caffe-action_recog/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

MODEL_FILE = 'examples/mine/net/net842.prototxt'
PRETRAINED = 'examples/mine/mine_train_iter_200000.caffemodel'
net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)

#set the input data
f=h5py.File("examples/mine/data/test.h5","r")
data=f['data']
label=f['label']
net.blobs['data'].reshape(data.size)
net.blobs['data'].reshape(599,1,8,1)#why can't be data.shape
net.blobs['data'].data[...]=data
net.blobs['label'].reshape(label.size)
net.blobs['label'].reshape(599,2)
net.blobs['label'].data[...]=label


"""
net.blobs['data'].data[...]=0
net.blobs['data'].reshape(8)
net.blobs['data'].data[...]=[0.84177828,0.84909189,0.83203268,0.68702471,0.26019111,0.92257178,0.28454423,0.08972001]#read from the test document and normalize the data
net.blobs['data'].reshape(1,1,8,1)
net.blobs['label'].reshape(1,2)#read from the test document
"""

#calculate the net's output
out=net.forward()#out equal to the loss
fc=net.blobs['fc3'].data[...]
label=net.blobs['label'].data

print "-------------error----------------"
batchsize=599
absmat=fc-label
absmat=abs(absmat)
relmat=absmat/label
absdif=sum(absmat,0)/batchsize
reldif=sum(relmat,0)/batchsize
print "absolute error:"
print absdif
print "relative error:"
print reldif

