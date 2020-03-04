"""
定义辅助函数
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly
#t-SNE可视化绘图函数
def plot_embedding(data,label,title):
    x_min,x_max=np.min(data,0),np.max(data,0)
    data=(data-x_min)/(x_max-x_min)

    fig=plt.figure()
    ax=plt.subplot(111)
    color=['b','c','g','k','m','r']
    colors=['rgba(88,87,86,1.0)','rgba(227,23,13,1.0)','rgba(255,192,203,1.0)','rgba(3,168,158,1.0)','rgba(227,207,87,1.0)','rgba(138,43,226,1.0)']
    #colors=['rgba('+str(r)+','+str(g)+','+str(b)+',1.0)' for r in np.linspace(25,168,6) for g in np.linspace(40,80,6) for b in np.linspace(0,255,6)]
    name=['buttercup','daisy','iris','lilyvalley','sunflower','windflower']
    Index=0
    PData=[]
    for co in color:
        Index=Index+1
        tempData=[]
        for i in range(data.shape[0]):
            if label[i]==Index:
                tempData.append(data[i,:])
        tempData=np.array(tempData)
        print(tempData.shape)
        trace=go.Scatter(
            x=tempData[:,0],
            y=tempData[:,1],
            name='{0}'.format(name[Index-1]),
            mode='markers',
            marker=dict(
                size=5,
                color=colors[Index-1]
            )
        )
        PData.append(trace)
        #plt.plot(data[i,0],data[i,1],color=color[label[i]-1],label="{0}".format(name[label[i]-1]))
        plt.plot(tempData[:,0],tempData[:,1],'o',color=co,label="{0}".format(name[Index-1]))
        plt.legend(numpoints=1)
    plt.xticks([])
    plt.yticks([])
    #plt.legend(numpoints=1)
    plt.title(title)

    layout=go.Layout(
        title='t-SNE visualization for texture features',
        yaxis=dict(zeroline=False),xaxis=dict(zeroline=False)
    )
    Pfig = go.Figure(data=PData, layout=layout)
    plotly.offline.plot(Pfig,filename='VB-TEXTURE-tSNE.html')
    return fig

#定义并计算卡方统计量距离
def chi2_distance(histA,histB,eps=1e-10):
    d=0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])
    return d

#图像分割函数
def segmentation(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _, _, imgV = cv2.split(imgHSV)
    m,n,c=imgHSV.shape
    imgV=np.array(imgV)/255
    imgVF=imgV.flatten()
    meanV=np.mean(imgVF,axis=0)
    varV=np.std(imgVF,axis=0)
    Threshold=meanV+varV
    fmask=np.zeros((m,n),dtype=np.uint8)
    mask=np.array(imgV>Threshold,dtype=np.uint8)
    img_segmentation=cv2.bitwise_and(img,img,mask=mask)
    return img_segmentation

#测试分割效果
if __name__=="__main__":
    filename='F:/seven/prDesign/test2.jpg'
    img=cv2.imread(filename)
    img_segmentation=segmentation(img)
    img_segmentation=cv2.resize(img_segmentation,(160,160))
    cv2.imwrite('F:/seven/prDesign/seg21.jpg', img_segmentation, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imshow('ori',img)
    cv2.imshow('seg',img_segmentation)
    cv2.waitKey(0)