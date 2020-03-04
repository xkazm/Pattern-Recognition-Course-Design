"""
加载数据模块
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import utils

#随机采样样本作为训练数据
def randomSample(imgNum,chosen_train):
    index=np.arange(imgNum)
    randomIndex=np.random.choice(index,size=chosen_train)
    setIndex=set(index)
    setRan=set(randomIndex)
    setresIndex=setIndex-setRan
    resIndex=np.array(list(setresIndex))
    return randomIndex,resIndex

#加载数据并作特征提取
def load(filenames,randomIndex,mode,chosen_train,n_points=0,radius=0):
    data=None
    num=np.zeros(6*chosen_train,dtype=int)
    k=0
    for filename in filenames:
        for i in range(chosen_train):
            tempImg=cv2.imread(filename[randomIndex[i]],cv2.IMREAD_COLOR)
            tempImg=utils.segmentation(tempImg)
            if mode=='shape':
                tempImg=cv2.cvtColor(tempImg,cv2.COLOR_BGR2GRAY)
                sift=cv2.xfeatures2d.SIFT_create()
                kps,des=sift.detectAndCompute(tempImg,None)
                num[k*chosen_train+i]=len(kps)
                tData=np.zeros((len(kps),5))
                for j in range(len(kps)):
                    tData[j,:]=np.array([kps[j].angle,kps[j].pt[0],kps[j].pt[1],kps[j].response,kps[j].size])[:]
            else:
                if mode=='color':
                    tempHSV=cv2.cvtColor(tempImg,cv2.COLOR_BGR2HSV)
                    m,n,c=tempHSV.shape
                    tempData = np.array(tempHSV)
                elif mode=='texture':
                    tempImg=cv2.cvtColor(tempImg,cv2.COLOR_BGR2GRAY)
                    tempLBP=local_binary_pattern(tempImg,n_points,radius)
                    m,n=tempLBP.shape
                    c=1
                    tempData=np.array(tempLBP).reshape((m,n,c))
                tempData=tempData.swapaxes(1,2)
                rowData=np.zeros((m,c*c+c))
                colData=np.zeros((n,c*c+c))
                for j in range(m):
                    rowData[j,:c*c]=np.cov(tempData[j,:,:]).reshape(-1)
                    rowData[j,c*c:]=np.mean(tempData[j,:,:],axis=1)
                for j in range(n):
                    colData[j,:c*c]=np.cov(np.mat(tempData[:,:,j]).T).reshape(-1)
                    colData[j,c*c:]=np.mean(tempData[:,:,j],axis=0)
                tData=np.concatenate((rowData,colData),axis=0)
            print(i)
            if data is None:
                data=tData
            else:
                data=np.concatenate((data,tData),axis=0)
        k=k+1
    return data,num