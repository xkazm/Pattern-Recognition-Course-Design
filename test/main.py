"""
方案一：基于视觉词汇的花卉识别方法
"""
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans,MiniBatchKMeans
import kNN
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold,datasets
import utils
import kNN
#import afkmc2.afkmc2 as afk
#import MR8
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data,io,data_dir,filters,feature
from skimage.color import label2rgb
import skimage
from PIL import Image
import dataLoad

CHOSEN_TRAIN=40#定义每一类选择的训练样本个数
RADIUS=4#定义LBP算子的区域半径
N_POINTS=8*RADIUS#定义LBP算子的区域采样点个数
ALPHA=6.0#颜色特征的权重
BETA=4.5#形状特征的权重
GAMMA=3.1#纹理特征的权重

randomIndex,resIndex=dataLoad.randomSample(80,CHOSEN_TRAIN)#随机选取训练样本和测试样本
filenames=[]
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/buttercup/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/daisy/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/iris/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/lilyvalley/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/sunflower/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/windflower/*.jpg'))
#提取原始特征
dataX,numX=dataLoad.load(filenames,randomIndex,'color',CHOSEN_TRAIN,N_POINTS,RADIUS)
print('color completed')
dataY,numY=dataLoad.load(filenames,randomIndex,'shape',CHOSEN_TRAIN,N_POINTS,RADIUS)
print('shape completed')
dataZ,numZ=dataLoad.load(filenames,randomIndex,'texture',CHOSEN_TRAIN,N_POINTS,RADIUS)
print('texture completed')
#生成类别标签
y=None
for i in range(6):
    tempY=(i+1)*np.ones(CHOSEN_TRAIN,dtype=int)
    if y is None:
        y=tempY
    else:
        y=np.concatenate((y,tempY),axis=0)

#聚类生成视觉词汇
kmeansX=MiniBatchKMeans(n_clusters=200)
kmeansX.fit(dataX)
print('color k-means completed')

kmeansY=MiniBatchKMeans(n_clusters=200)
kmeansY.fit(dataY)
print('shape k-means completed')

kmeansZ=MiniBatchKMeans(n_clusters=200)
kmeansZ.fit(dataZ)
print('texture k-means completed')
print('k-means complete!')
centersX,labelsX=kmeansX.cluster_centers_,kmeansX.labels_
centersY,labelsY=kmeansY.cluster_centers_,kmeansY.labels_
centersZ,labelsZ=kmeansZ.cluster_centers_,kmeansZ.labels_

#生成最终的特征表达，即用最近邻法统计直方图
histDataX=np.zeros((6*CHOSEN_TRAIN,200),dtype=int)
histDataY=np.zeros((6*CHOSEN_TRAIN,200),dtype=int)
histDataZ=np.zeros((6*CHOSEN_TRAIN,200),dtype=int)
k=0
sumXZ=0
sumY=0
for filename in filenames:
    for i in range(CHOSEN_TRAIN):
        tempImg=cv2.cvtColor(cv2.imread(filename[i],cv2.IMREAD_COLOR),cv2.COLOR_BGR2HSV)        #m,n,_=tempImg.shape
        m,n,_=tempImg.shape
        for j in range(m+n):
            histDataX[k*CHOSEN_TRAIN+i,labelsX[j+sumXZ]]=histDataX[k*CHOSEN_TRAIN+i,labelsX[j+sumXZ]]+1
            histDataZ[k*CHOSEN_TRAIN+i,labelsZ[j+sumXZ]]=histDataZ[k*CHOSEN_TRAIN+i,labelsZ[j+sumXZ]]+1
        for j in range(numY[k*CHOSEN_TRAIN+i]):
            histDataY[k * CHOSEN_TRAIN + i, labelsY[j + sumY]] = histDataY[k * CHOSEN_TRAIN + i, labelsY[j + sumY]] + 1
        sumXZ=sumXZ+m+n
        sumY=sumY+numY[k*CHOSEN_TRAIN+i]
    k=k+1


#"""
#使用t-SNE数据可视化
tsne=manifold.TSNE(n_components=2,init='pca',random_state=0)
Y=tsne.fit_transform(histDataY)
print('tSNE completed!')
fig=utils.plot_embedding(Y,y,'t-SNE embedding of the digits')
plt.show()
#"""
#测试，并计算正确率
"""
sum=0
t=0
for filename in filenames:
    t=t+1
    for i in range(CHOSEN_TRAIN,80):
        resultTT=np.zeros(7,dtype=float)
        testFile=filename[resIndex[i-CHOSEN_TRAIN]]
        histTestX=kNN.featureCOV(centersX,testFile)
        histTestY=kNN.featureSIFT(centersY,testFile)
        histTestZ=kNN.featureLBP(centersZ,testFile,N_POINTS,RADIUS)
        print('featuren construction completed!')
        resultC,probC=kNN.xkNN(histDataX,y,histTestX)
        resultS,probS=kNN.xkNN(histDataY,y,histTestY)
        resultT,probT=kNN.xkNN(histDataZ,y,histTestZ)
        resultTT[resultC]=resultTT[resultC]+ALPHA*probC
        resultTT[resultS]=resultTT[resultS]+BETA*probS
        resultTT[resultT]=resultTT[resultT]+GAMMA*probT
        result=np.where(resultTT==np.max(resultTT))
        result=result[0]
        print(resultTT,result,t)
        if result==t:
            sum=sum+1
print(sum*100/240)
"""
#测试函数
testFilename='F:/seven/prDesign/test6.jpg'
resultTT=np.zeros(7,dtype=float)
histTestX=kNN.featureCOV(centersX,testFilename)
histTestY=kNN.featureSIFT(centersY,testFilename)
histTestZ=kNN.featureLBP(centersZ,testFilename,N_POINTS,RADIUS)
print('featuren construction completed!')
resultC,probC=kNN.xkNN(histDataX,y,histTestX)
resultS,probS=kNN.xkNN(histDataY,y,histTestY)
resultT,probT=kNN.xkNN(histDataZ,y,histTestZ)
resultTT[resultC]=resultTT[resultC]+ALPHA*probC
resultTT[resultS]=resultTT[resultS]+BETA*probS
resultTT[resultT]=resultTT[resultT]+GAMMA*probT
result=np.where(resultTT==np.max(resultTT))
result=result[0]
print(resultTT,result)