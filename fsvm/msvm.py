"""
SVM分类函数模块
"""
import numpy
import sklearn.svm as svm
import numpy as np

#随机分割训练集和测试集
def randomSample(imgNum,chosen_train):
    index=np.arange(imgNum)
    #print(index)
    randomIndex=np.random.choice(index,size=chosen_train,replace=False)
    setIndex=set(index)
    setRan=set(randomIndex)
    setresIndex=setIndex-setRan
    resIndex=np.array(list(setresIndex))
    print(randomIndex)
    print(resIndex)
    return randomIndex,resIndex

#定义核函数
def myKernel(x,y):
    sum=0
    for i in range(len(x)):
        if x[i]<y[i]:
            sum=sum+x[i]
        else:
            sum=sum+y[i]
    return sum
#计算核函数
def toKernel(X,Y):
    n=len(X)
    m=len(Y)
    result=np.zeros((n,m))
    for i in range(n):
        print(i)
        for j in range(m):
           result[i][j]=myKernel(X[i],Y[j])
    return result
#svm分类
def svmClassication(X,y):
    wsvm=svm.SVC(kernel='precomputed')
    XN=toKernel(X,X)
    wsvm.fit(XN,y)
    return wsvm