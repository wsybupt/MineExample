# MineExample
使用caffe解决非图像问题的回归问题的一个例子。
## 文件列表
###data/
hdf5数据文件和用于在solver中指定数据位置的text文件。
###raw data/
原始数据，从raw data.xlsx文件中可以看到数据的含义。
###net/
网络配置的prototxt文件,文件名注明训练网络各层的神经元个数。
###solver/
训练的solver文件,与net文件对应
###writemine.py
对数据进行归一化等预处理，写入hdf5数据,运行后生成3个文件:
```
mine.h5:包含全部1599个数据
train.h5：包含训练集的1000个数据
test.h5：包含测试集的599个数据
```
###demo.py
输入测试集数据计算预测值与实际值的相对误差。

##运行方法
###训练网络：
sudo ./build/tools/caffe train --solver examples/mine/solver/solver.prototxt 

##测试方法
在caffe根目录下运行
```
sudo python a.py
sudo python demo.py
```
a.py测试1000个训练数据，demo.py测试599个测试数据。

不同的网络结果下需要修改的包括：

1.网络结构
```
  MODEL_FILE = 'examples/mine/net/net.prototxt'
```
2.caffemodel
```
  PRETRAINED = 'examples/mine/mine_train_iter_200000.caffemodel'
```
3.读出数据的位置
```
  fc=net.blobs['fc4'].data
```



# MineExample
Use caffe to handle the regression problem and the input data is not image.

##Document list
###data/
The hdf5 data and the text file used for training solver prototxt.
###row data/
The row data in .xlsx format and text format. You can find the data's meaning in the xlsx file.
###net/
The prototxt document that describes the net. From the name you can see how many layer are there in the net and 
how many neurons in each layer.
###solver/
The solver document for each net prototxt.
###writemine.py
Create the hdf5 data from the file in the 'row data' folder. After running it, there will be 3 hdf5 data file:
```
  mine.h5, which includes 1599 sample;
  train.h5, which includes 1000 samplea as training set;
  test.h5, which includes 599 samples as testing set.
```
