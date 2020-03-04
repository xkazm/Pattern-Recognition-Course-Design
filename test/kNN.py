"""
KNN分类模块
"""
import numpy as np
import utils
import cv2
import MR8
from skimage.feature import local_binary_pattern

#单一特征的KNN分类函数
def xkNN(data,y,testX,k=5):
    m,n=data.shape
    disMatrix=np.zeros(m,dtype=float)
    for i in range(m):
        disMatrix[i]=utils.chi2_distance(data[i],testX)
    disIndex=np.argsort(disMatrix,axis=0)
    disIndexR=disIndex[:k]
    result=y[disIndexR]
    bresult=np.bincount(result)
    print(bresult)
    resultLabel=np.argmax(bresult)
    sum=np.sum(bresult)
    prob=float(bresult[resultLabel])/float(sum)
    return resultLabel,prob

#联合特征的KNN分类函数
def bxNN(dataX,dataY,dataZ,y,testX,testY,testZ,alpha,beta,gamma,k=5):
    mx,nx=dataX.shape
    my,ny=dataY.shape
    mz,nz=dataZ.shape
    disXMatrix=np.zeros(mx,dtype=float)
    disYMatrix=np.zeros(my,dtype=float)
    disZMatrix=np.zeros(mz,dtype=float)
    for i in range(mx):
        disXMatrix[i]=utils.chi2_distance(dataX[i],testX)
    for i in range(my):
        disYMatrix[i]=utils.chi2_distance(dataY[i],testY)
    for i in range(mz):
        disZMatrix[i]=utils.chi2_distance(dataZ[i],testZ)
    disMatrix=alpha*disXMatrix+beta*disYMatrix+gamma*disZMatrix
    disIndex = np.argsort(disMatrix, axis=0)
    disIndexR = disIndex[:k]
    result = y[disIndexR]
    print(np.bincount(result))
    resultLabel = np.argmax(np.bincount(result))
    return resultLabel

#最近邻分类函数
def ekNN(data,y,testX,k=1):
    m,n=data.shape
    disMatrix=np.zeros(m,dtype=float)
    for i in range(m):
        disMatrix[i]=np.linalg.norm(data[i,:]-testX)
    disIndex=np.argsort(disMatrix,axis=0)
    disIndexR=disIndex[:k]
    result=y[disIndexR]
    return result

#提取颜色特征
def featureCOV(center,filename):
    img=cv2.imread(filename,cv2.IMREAD_COLOR)
    img=utils.segmentation(img)
    hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cluster=len(center)
    m,n,c=hsvImg.shape
    data=np.array(hsvImg)
    data=data.swapaxes(1,2)
    rowData = np.zeros((m, c * c + c))
    colData = np.zeros((n, c * c + c))
    for j in range(m):
        rowData[j, :c * c] = np.cov(data[j, :, :]).reshape(-1)
        rowData[j, c * c:] = np.mean(data[j, :, :], axis=1)
        ttData = np.sort(data[j, :, :], axis=1)
    for j in range(n):
        colData[j, :c * c] = np.cov(np.mat(data[:, :, j]).T).reshape(-1)
        colData[j, c * c:] = np.mean(data[:, :, j], axis=0)
    tData = np.concatenate((rowData, colData), axis=0)
    y=np.arange(cluster)+1
    histFeature=np.zeros(cluster)
    for i in range(m+n):
        tempLabel=ekNN(center,y,tData[i])
        histFeature[tempLabel-1]=histFeature[tempLabel-1]+1
    return histFeature

#提取形状特征
def featureSIFT(center,filename):
    tempImg = cv2.imread(filename, cv2.IMREAD_COLOR)
    tempImg=utils.segmentation(tempImg)
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
    data=None
    cluster=len(center)
    sift = cv2.xfeatures2d.SIFT_create()
    kps, des = sift.detectAndCompute(tempImg, None)
    # print(kps)
    tData = np.zeros((len(kps), 5))
    for j in range(len(kps)):
        tData[j, :] = np.array([kps[j].angle, kps[j].pt[0], kps[j].pt[1], kps[j].response, kps[j].size])[:]
    if data is None:
        data = tData
    else:
        data = np.concatenate((data, tData), axis=0)

    y = np.arange(cluster) + 1
    histFeature = np.zeros(cluster)
    for i in range(len(kps)):
        tempLabel = ekNN(center, y, tData[i])
        histFeature[tempLabel - 1] = histFeature[tempLabel - 1] + 1
    return histFeature

#提取纹理特征
def featureLBP(center,filename,n_points,radius):
    tempImg = cv2.imread(filename, cv2.IMREAD_COLOR)
    tempImg=utils.segmentation(tempImg)
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
    tempLBP=local_binary_pattern(tempImg,n_points,radius)
    m, n = tempLBP.shape
    c=1
    tempData = np.array(tempLBP).reshape((m,n,c))
    tempData = tempData.swapaxes(1, 2)
    rowData = np.zeros((m, c * c + c))
    colData = np.zeros((n, c * c + c))
    data = None
    cluster = len(center)
    for j in range(m):
        rowData[j, :c * c] = np.cov(tempData[j, :, :]).reshape(-1)
        rowData[j, c * c:] = np.mean(tempData[j, :, :], axis=1)
    for j in range(n):
        colData[j, :c * c] = np.cov(np.mat(tempData[:, :, j]).T).reshape(-1)
        colData[j, c * c:] = np.mean(tempData[:, :, j], axis=0)
    tData = np.concatenate((rowData, colData), axis=0)
    if data is None:
        data = tData
    else:
        data = np.concatenate((data, tData), axis=0)

    y = np.arange(cluster) + 1
    histFeature = np.zeros(cluster)
    for i in range(m + n):
        tempLabel = ekNN(center, y, tData[i])
        histFeature[tempLabel - 1] = histFeature[tempLabel - 1] + 1
    return histFeature