"""
Sparse-SIFT特征提取模块和图像分割模块
"""
import numpy as np
import sSIFT
import ksvd
import cv2
import glob

#基于OTSU方法的图像分割函数
def segment(img):
    uimg=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    udata=uimg[:,:,1]
    Lmax=np.max(udata)
    muT=np.mean(np.mean(udata,axis=0))
    Object=0
    T=0
    T,mask=cv2.threshold(udata,0,Lmax,cv2.THRESH_OTSU)
    img_segmentation=cv2.bitwise_and(img,img,mask=mask)
    return img_segmentation,mask

#稀疏编码的SIFT特征提取函数
def SIFT(data):
    n=len(data)
    sift_feature=np.zeros((n,128))
    detector=sSIFT.DenseSIFT()
    for i in range(n):
        temp_feature=detector.detectAndCompute(data[i,:,:])
        print(i,temp_feature.shape)
        skvd=ksvd.KSVD(2720)
        dictionary,sparse_code=skvd.fit(temp_feature)
        sift_feature[i]=np.mean(sparse_code,axis=0).astype(float)
    return sift_feature

#测试函数
if __name__=="__main__":
    img = cv2.imread("F:/seven/prDesign/test3.jpg")
    img_re=cv2.resize(img,(160,160))
    """
    filenames = sorted(glob.glob(r'F:/seven/prDesign/oxford102/segmim/*.jpg'))
    root='F:/seven/prDesign/oxford102/mask/'
    for m in range(8189):
        img=cv2.imread(filenames[m])
        print(filenames[m])
        mask=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,0]==254 and img[i,j,1]==0 and img[i,j,2]==0:
                    mask[i][j]=0
                else:
                    mask[i][j]=255
        mask_file=root+str(m+1)+'.jpg'
        cv2.imwrite(mask_file, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    """
    cv2.imshow('origin',img)
    img_segmentation,maskk=segment(img)
    img_segmentation=cv2.resize(img_segmentation,(160,160))
    cv2.imwrite('F:/seven/prDesign/seg32.jpg', img_segmentation, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imshow('segmented',img_segmentation)
    #cv2.imshow('mask',mask)
    cv2.waitKey(0)