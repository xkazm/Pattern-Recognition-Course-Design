"""
SFTA特征提取模块
"""
import numpy as np
import cv2
import math

def otsu(counts):
    """
    通过OSTU算法获得阈值
    :param counts: 颜色直方图
    :return: 最佳阈值
    """
    p=counts/sum(counts)#归一化
    Tbin=0#定义起始值
    ToBin=len(p)#定义结束值
    Object=0
    x=np.arange(ToBin)
    meanT=0
    #求直方图中颜色的加权平均
    for i in range(ToBin):
        meanT=meanT+p[i]*x[i]
    #寻找最佳阈值
    for t in range(ToBin):
        mean0=0
        mean1=0
        #背景的颜色均值
        for i in range(t):
            mean0=mean0+p[i]*x[i]
        #目标的颜色均值
        for i in range(t,ToBin):
            mean1=mean1+p[i]*x[i]
        w0=np.sum(p[:t])
        w1=np.sum(p[t:])
        Nobject=w0*(mean0-meanT)**2+w1*(mean1-meanT)**2#最大化类间差异
        if Nobject>Object:
            Object=Nobject
            Tbin=t
    return Tbin

def recur_ostu(T,counts,lowerBin,upperBin,tLower,tUpper):
    """
    利用递归的方式求解多等级阈值分割的阈值集合
    :param T: 阈值集合
    :param counts: 颜色直方图
    :param lowerBin: 阈值搜寻下界
    :param upperBin: 阈值搜寻上界
    :param tLower: 阈值的位置下界
    :param tUpper: 阈值的位置上界
    """
    if tUpper<tLower or lowerBin>=upperBin:
        return
    else:
        thres=otsu(counts[lowerBin:upperBin])+lowerBin
        thPos=int((tLower+tUpper)/2)#确定所求得的阈值对应在阈值集合中的位置
        T[thPos]=thres
        recur_ostu(T,counts,lowerBin,thres,tLower,thPos-1)
        recur_ostu(T,counts,thres+1,upperBin,thPos+1,tUpper)

def multiOTSU(img,nt):
    """
    多等级阈值求解
    :param img: 输入图像
    :param nt: 阈值的个数
    :return: 阈值集合
    """
    img.astype(np.uint8)
    counts=cv2.calcHist([img],[0],None,[256],[0,256])
    T=np.zeros(nt)
    recur_ostu(T,counts,0,255,0,nt-1)
    return T

def hausDim(img):
    """
    计算分形维度
    :param img:输入图像
    :return: 分形维度
    """
    m,n=img.shape
    maxDim=np.max([m,n])
    newDimSize=int(math.pow(2,int(math.log2(maxDim))+1))
    rowPad=newDimSize-m
    colPad=newDimSize-n
    I=np.lib.pad(img,((rowPad,0),(colPad,0)),'symmetric')#补全图像至长、宽均为2的整数次幂

    newm,newn=I.shape

    boxCounts=np.zeros(int(math.log2(maxDim))+1)
    resolutions=np.zeros(int(math.log2(maxDim))+1)

    m,n=I.shape
    boxSize=m#初始box的大小
    boxesPerDim=1#初始box的个数
    idx=0
    #不同的box大小统计目标点的个数
    while boxSize>1:
        boxCount=0
        minBox=np.arange(0,(m-boxSize)+1,boxSize).astype(np.int)
        maxBox=np.arange(boxSize-1,m,boxSize).astype(np.int)

        for boxRow in range(boxesPerDim):
            for boxCol in range(boxesPerDim):
                objFound=False
                for row in range(minBox[boxRow],maxBox[boxRow]):
                    for col in range(minBox[boxCol],maxBox[boxCol]):
                        if I[row,col]:
                            boxCount=boxCount+1
                            objFound=True
                            break
                    if objFound:
                        break

        boxCounts[idx]=boxCount+1e-5#防止为0，对对数求解造成影响
        resolutions[idx]=float(1/boxSize)+1e-5
        idx=idx+1

        boxesPerDim=boxesPerDim*2
        boxSize=int(boxSize)/2

    D=np.polyfit(np.log(resolutions),np.log(boxCounts),1)#线性拟合，返回斜率
    return D[0]

def sfta(img,nt):
    """
    SFTA特征提取函数
    :param img: 输入图像
    :param nt: 阈值个数
    :return: SFTA特征向量
    """
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayImg.astype(np.uint8)
    T=multiOTSU(grayImg,nt)
    dSize=(len(T)*6)-3
    D=np.zeros(dSize)
    pos=0
    #计算单阈值情形
    for t in range(len(T)):
        thres=T[t]

        _,Ib=cv2.threshold(grayImg,thres,255,cv2.THRESH_BINARY)

        edge_output=cv2.Canny(Ib,threshold1=50,threshold2=150)
        Ib=cv2.bitwise_and(Ib,Ib,mask=edge_output)

        IIb=cv2.bitwise_and(grayImg,grayImg,mask=edge_output)
        Val=IIb.ravel()[np.flatnonzero(IIb)]

        D[pos]=hausDim(Ib)#分形维度
        pos=pos+1

        D[pos]=np.mean(np.mean(Val)) if len(Val)!=0 else 0#平均灰度
        pos=pos+1

        D[pos]=len(Val)#目标个数
        pos=pos+1
    #两阈值分割的情形
    for t in range(len(T)-1):
        lowerThres=T[t]
        upperThres=T[t+1]

        #print(lowerThres,upperThres)

        Ibu=np.array(grayImg<upperThres,dtype=np.uint8)
        Ibl=np.array(grayImg>lowerThres,dtype=np.uint8)

        Ib=Ibu*Ibl*255
        Ib.astype(np.uint8)

        edge_output = cv2.Canny(Ib, threshold1=50, threshold2=150)
        Ib = cv2.bitwise_and(Ib, Ib, mask=edge_output)

        IIb = cv2.bitwise_and(grayImg, grayImg, mask=edge_output)
        Val = IIb.ravel()[np.flatnonzero(IIb)]

        D[pos] = hausDim(Ib)
        pos = pos + 1

        D[pos] = np.mean(np.mean(Val)) if len(Val)!=0 else 0
        pos = pos + 1

        D[pos] = len(Val)
        pos = pos + 1

    return D

#测试函数
if __name__=="__main__":
    img = cv2.imread("F:/seven/prDesign/fdata/buttercup/image_0021.jpg")
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    T=multiOTSU(grayImg,10)
    D=sfta(img,10)
    print(D)