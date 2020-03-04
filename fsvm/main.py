"""
方案二的主程序模块
"""
import glob
import cv2
import numpy as np
import extractionFeature as exF
import msvm
import sklearn.svm as svm
from sklearn.cluster import KMeans
from sklearn import manifold
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly
from sklearn.metrics import plot_confusion_matrix

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
        title='t-SNE visualization for Fusion features',
        yaxis=dict(zeroline=False),xaxis=dict(zeroline=False)
    )
    Pfig = go.Figure(data=PData, layout=layout)
    plotly.offline.plot(Pfig,filename='VB-FUSION-tSNE.html')
    return fig

#加载数据
filenames=[]
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/buttercup/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/daisy/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/iris/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/lilyvalley/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/sunflower/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/windflower/*.jpg'))
#定义存储提取好的特征向量的文件路径
root="F:/seven/prDesign/data/"
tail=".txt"
HCB=[]
SCB=[]
VCB=[]
GCB=[]
ECB=[]
HC=[]
SC=[]
VC=[]
GC=[]
EC=[]
for i in range(21):
    HCB.append(root+"HCB"+str(i+1)+tail)
    SCB.append(root+"SCB"+str(i+1)+tail)
    VCB.append(root+"VCB"+str(i+1)+tail)
    GCB.append(root+"GCB"+str(i+1)+tail)
    ECB.append(root+"ECB"+str(i+1)+tail)
    HC.append(root+"HC"+str(i+1)+tail)
    SC.append(root+"SC"+str(i+1)+tail)
    VC.append(root+"VC"+str(i+1)+tail)
    GC.append(root+"GC"+str(i+1)+tail)
    EC.append(root+"EC"+str(i+1)+tail)

"""
#生成特征并存储
for i in range(21):
    if i==0:
        level=0
        split=False
        split_op=0
    elif i>0 and i<5:
        level=1
        split=True
        split_op=i-1
    else:
        level=2
        split=True
        split_op=i-5
    print('level:split_op:',level,split_op)
    exF.saveHSVcode(filenames,HCB[i],SCB[i],VCB[i],HC[i],SC[i],VC[i],split,split_op,level)
    print('Hsv')
    exF.saveGcode(filenames,GCB[i],GC[i],split,split_op,level)
    print('Gray')
    exF.saveEcode(filenames,ECB[i],EC[i],split,split_op,level)
    print('Edge')
"""
#读取已经生成好的特征并融合
FCode=np.zeros((480,21*300))
for i in range(21):
    tempHCode=np.loadtxt(HC[i])
    #tempHCode=np.nan_to_num(tempHCode)
    FCode[:,180*i:180*i+60]=tempHCode
    tempSCode=np.loadtxt(SC[i])
    #tempSCode=np.nan_to_sum(tempSCode)
    FCode[:,180*i+60:180*i+120]=tempSCode
    tempVCode=np.loadtxt(VC[i])
    FCode[:,180*i+120:180*i+180]=tempVCode
    tempGCode=np.loadtxt(GC[i])
    FCode[:,3780+60*i:3840+60*i]=tempGCode
    tempECode=np.loadtxt(EC[i])
    FCode[:,5040+60*i:5100+60*i]=tempECode
FCode=np.nan_to_num(FCode)
print(FCode.shape)
print(FCode[0])

imgNum=80
chosen_train=40
#定义标签
y=None
for i in range(6):
    tempY=(i+1)*np.ones(chosen_train,dtype=int)
    if y is None:
        y=tempY
    else:
        y=np.concatenate((y,tempY),axis=0)
#生成训练样本和测试样本
Ftrain=np.zeros((6*40,6300))
Ftest=np.zeros((6*40,6300))
ytrain=y[:]
ytest=y[:]

yH=np.concatenate((y,y),axis=0)
#"""
#使用t-SNE数据可视化
tsne=manifold.TSNE(n_components=2,init='pca',random_state=0)
Y=tsne.fit_transform(FCode)
print('tSNE completed!')
fig=plot_embedding(Y,yH,'t-SNE embedding of the digits')
plt.show()
#"""

for i in range(6):
    print(i)
    randomIndex,resIndex=msvm.randomSample(imgNum,chosen_train)
    print(len(randomIndex),len(resIndex))
    Ftrain[40*i:40*(i+1),:]=FCode[randomIndex+80*i,:]
    Ftest[40*i:40*(i+1),:]=FCode[resIndex+80*i,:]
#分类和测试
isvm=msvm.svmClassication(Ftrain,ytrain)
#isvm=svm.SVC(kernel='rbf')
#isvm.fit(Ftrain,ytrain)
FNtest=msvm.toKernel(Ftest,Ftrain)
score=isvm.score(FNtest,ytest)
print(score)

#分类测试示例
testFile='F:/seven/prDesign/test6.jpg'
img=cv2.imread(testFile)
img=cv2.resize(img,(300,300))
tHCode=np.zeros((21,60))
tSCode=np.zeros((21,60))
tVCode=np.zeros((21,60))
tGCode=np.zeros((21,60))
tECode=np.zeros((21,60))
for i in range(21):
    if i==0:
        level=0
        split=False
        split_op=0
    elif i>0 and i<5:
        level=1
        split=True
        split_op=i-1
    else:
        level=2
        split=True
        split_op=i-5
    print('level:split_op:',level,split_op)
    tHCode[i],tSCode[i],tVCode[i]=exF.PHOW_HSV(img,HCB[i],SCB[i],VCB[i],split,split_op,level)
    print('Hsv')
    tGCode[i]=exF.PHOW_G(img,GCB[i],split,split_op,level)
    print('Gray')
    tECode[i]=exF.PHOW_E(img,ECB[i],split,split_op,level)
    print('Edge')
tFCode=np.zeros(6300)
for i in range(21):
    tFCode[180*i:180*i+60]=tHCode[i]
    tFCode[180*i+60:180*i+120]=tSCode[i]
    tFCode[180*i+120:180*i+180]=tVCode[i]
    tFCode[3780+60*i:3840+60*i]=tGCode[i]
    tFCode[5040+60*i:5100+60*i]=tECode[i]
tFCode=np.nan_to_num(tFCode)
print(tFCode.shape)
tFNtest=msvm.toKernel(tFCode.reshape(1,-1),Ftrain)
cls=isvm.predict(tFNtest)
print(cls)
np.set_printoptions(precision=2)
name=['buttercup','daisy','iris','lilyvalley','sunflower','windflower']
# Plot non-normalized confusion matrix
titles_options = [("Normalized confusion matrix for SVM-HIK", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(isvm, FNtest, y,
                                 display_labels=name,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

