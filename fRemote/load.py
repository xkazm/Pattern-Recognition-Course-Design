import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly
import plotly.graph_objs as go

def plot_embedding(data,label,title):
    x_min,x_max=np.min(data,0),np.max(data,0)
    data=(data-x_min)/(x_max-x_min)

    fig=plt.figure()
    ax=plt.subplot(111)
    color=['b','c','g','k','m','r']
    colors=['rgba(88,87,86,1.0)','rgba(227,23,13,1.0)','rgba(255,192,203,1.0)','rgba(3,168,158,1.0)','rgba(227,207,87,1.0)','rgba(138,43,226,1.0)']
    #colors=['rgba('+str(r)+','+str(g)+','+str(b)+',1.0)' for r in np.linspace(25,168,6) for g in np.linspace(40,80,6) for b in np.linspace(0,255,6)]
    name=['buttercup','daisy','iris','lilyvalley','sunflower','windflower']
    Index=-1
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
        title='t-SNE visualization for Inception-V3 features',
        yaxis=dict(zeroline=False),xaxis=dict(zeroline=False)
    )
    Pfig = go.Figure(data=PData, layout=layout)
    plotly.offline.plot(Pfig,filename='VB-TEXTURE-tSNE.html')
    return fig

M=np.array([
    [0.412453,0.357580,0.180423],
    [0.212671,0.715160,0.072169],
    [0.019334,0.119193,0.950227]
])

def f(im_channel):
    return np.power(im_channel,1/3) if im_channel>0.008856 else 7.787*im_channel+0.137931

def __rgb2xyz__(pixel):
    b,g,r=pixel[2],pixel[1],pixel[0]
    rgb=np.array([r,g,b])
    XYZ=np.dot(M,rgb.T)
    XYZ=XYZ/255.0
    return (XYZ[0]/0.95047,XYZ[1]/1.0,XYZ[2]/1.0883)

def __xyz2lab__(xyz):
    F_XYZ=[f(x) for x in xyz]
    L=116*F_XYZ[1]-16 if xyz[1]>0.008856 else 903.3*xyz[1]
    a=500*(F_XYZ[0]-F_XYZ[1])
    b=200*(F_XYZ[0]-F_XYZ[2])
    return L,a,b

def RGB2Lab(pixel):
    xyz=__rgb2xyz__(pixel)
    L,a,b=__xyz2lab__(xyz)
    return L,a,b

def HisSegmentation(img):
    uImg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    lImg=cv2.cvtColor(uImg,cv2.COLOR_RGB2Lab)
    Hiscenter=np.array([11,33,55,77,99,121,143,165,187,209,231,249])
    valueThre=22
    Hisc=np.zeros((12*12*12,3))
    LHisc=np.zeros((12*12*12,3))
    for i in range(12):
        for j in range(12):
            for k in range(12):
                Hisc[i*12*12+j*12+k,0]=Hiscenter[i]
                Hisc[i*12*12+j*12+k,1]=Hiscenter[j]
                Hisc[i*12*12+j*12+k,2]=Hiscenter[k]
                LHisc[i*12*12+j*12+k,0],LHisc[i*12*12+j*12+k,1],LHisc[i*12*12+j*12+k,2]=RGB2Lab(Hisc[i*12*12+j*12+k,:])
    LHisc=LHisc.reshape(12*12*12,3)
    Histo=np.zeros(12*12*12)
    m,n,_=uImg.shape
    for i in range(m):
        for j in range(n):
            index=int(uImg[i,j,0]/22)*12*12+int(uImg[i,j,1]/22)*12+int(uImg[i,j,2]/22)
            Histo[index]=Histo[index]+1
    #print(Histo)
    sortIndex=np.argsort(-Histo,axis=0)
    Histo=-np.sort(-Histo,axis=0)
    print(Histo)
    endIndex=0
    sumPixels=0
    thres=m*n*0.95
    #print(len(index))
    for i in range(12*12*12):
        sumPixels=sumPixels+Histo[i]
        if sumPixels>=thres:
            endIndex=i
            break
    index=sortIndex[:endIndex]
    print(len(index))
    NLhisc=np.zeros((len(index),3))
    for i in range(len(index)):
        NLhisc[i]=LHisc[index[i]]
    SLhisc=np.zeros(len(index))
    for i in range(len(index)):
        sum=0.
        c_l=NLhisc[i,:]
        for j in range(len(index)):
            disc=np.linalg.norm(c_l-NLhisc[j,:])
            sum=sum+(float(Histo[j])/(m*n))*disc
        SLhisc[i]=sum
    mm=int(len(index)/4)
    result=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            tempData=uImg[i,j,:]
            inD=int(tempData[0]/22)*12*12+int(tempData[1]/22)*12+int(tempData[2]/22)
            if inD in index:
                tinD=np.where(index==inD)
                result[i][j]=SLhisc[tinD]
            else:
                tempLData=lImg[i,j,:]
                dist=np.zeros(len(index))
                for k in range(len(index)):
                    dist[k]=np.linalg.norm(tempLData-NLhisc[k,:])
                LIndex=np.argsort(dist,axis=0)
                LIndex=LIndex[:mm]
                mdist=np.sort(dist,axis=0)[:mm]
                t=np.sum(mdist)
                sum=0
                for k in LIndex:
                    sum=sum+(t-dist[k])*SLhisc[k]
                result[i][j]=sum/((mm-1)*t)
    max=np.max(np.max(result,axis=0),axis=0)
    min=np.min(np.min(result,axis=0),axis=0)
    print(max,min)
    print(result[1][2])
    result=255*(result-min)/(max-min)
    print(result[1][2])
    return result

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

def load(filenames):
    data=[]
    label=[]
    k=0
    for filename in filenames:
        for i in range(len(filename)):
            tempData=cv2.imread(filename[i],cv2.IMREAD_COLOR)
            tempData=segmentation(tempData)
            tempData=cv2.cvtColor(tempData,cv2.COLOR_BGR2HSV)
            tempData=cv2.resize(tempData,(299,299))
            tempData=np.swapaxes(tempData,1,2)
            tempData=np.swapaxes(tempData,0,1)
            tempData=tempData/255.
            data.append(tempData)
            label.append(k)
        k=k+1
    return np.array(data,dtype=float),np.array(label)

if __name__=="__main__":
    """
    img=cv2.imread('F:/seven/prDesign/fdata/buttercup/image_0009.jpg')
    imgS=HisSegmentation(img)
    imgS=imgS.astype(np.uint8)
    #ret, imgS = cv2.threshold(imgS, 100, 255, cv2.THRESH_BINARY)
    print(imgS)
    cv2.imshow('seg',imgS)
    cv2.waitKey(0)
    """
    # img = cv2.imread("F:/seven/prDesign/oxford102/segmim/segmim_00036.jpg")
    filenames = sorted(glob.glob(r'oxford102/segmim/*.jpg'))
    root = 'oxford102/mask/'
    for m in range(8189):
        img = cv2.imread(filenames[m])
        print(filenames[m])
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j, 0] == 254 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                    mask[i][j] = 0
                else:
                    mask[i][j] = 255
        mask_file = root + str(m + 1) + '.jpg'
        cv2.imwrite(mask_file, mask, [int(cv2.IMWRITE_JPEG_QUALITY), 70])