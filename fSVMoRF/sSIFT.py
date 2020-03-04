"""
Dense-SIFT模块
"""
import numpy as np
import cv2

#定义DenseSIFT特征提取算子
class DenseSIFT():
    def __init__(self):
        self.sift=cv2.xfeatures2d.SIFT_create()

    def detectAndCompute(self,image,step_size=8,window_size=(16,16)):
        if window_size is None:
            winH,winW=image.shape[:2]
            window_size=(winW//4,winH//4)

        descriptors=np.array([],dtype=np.float32).reshape(0,128)
        for crop in self._crop_image(image,step_size,window_size):
            tmp_descriptor=self._detectAndCompute(crop)[1]
            if tmp_descriptor is None:
                continue
            descriptors=np.vstack([descriptors,tmp_descriptor])
        return descriptors

    def _detect(self,image):
        return self.sift.detect(image)

    def _compute(self,image,kps,eps=1e-7):
        kps,descs=self.sift.compute(image,kps)

        if len(kps)==0:
            return [],None

        descs/=(descs.sum(axis=1,keepdims=True)+eps)
        descs=np.sqrt(descs)
        return kps,descs

    def _detectAndCompute(self,image):
        kps=self._detect(image)
        return self._compute(image,kps)

    def _sliding_window(self,image,step_size,window_size):
        for y in range(0,image.shape[0],step_size):
            for x in range(0,image.shape[1],step_size):
                yield (x,y,image[y:y+window_size[1],x:x+window_size[0]])

    def _crop_image(self,image,step_size=8,window_size=(16,16)):
        crops=[]
        winH,winW=window_size
        for (x,y,window) in self._sliding_window(image,step_size=step_size,window_size=window_size):
            if window.shape[0]!=winH or window.shape[1]!=winW:
                continue
            crops.append(image[y:y+winH,x:x+winW])
        return np.array(crops)

#测试函数
if __name__=="__main__":
    img=cv2.imread("F:/seven/prDesign/fdata/iris/image_0401.jpg")
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Densift=DenseSIFT()
    des1=Densift.detectAndCompute(imgHSV[:,:,0])
    des2=Densift.detectAndCompute(imgHSV[:,:,1])
    des3=Densift.detectAndCompute(imgHSV[:,:,2])
    print(des1.shape)
    print(des2.shape)
    print(des3.shape)