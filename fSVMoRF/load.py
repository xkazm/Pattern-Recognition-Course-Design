"""
数据加载模块
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sfta
import featureExtract as fE
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
            if label[i]==Index-1:
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
        title='t-SNE visualization for SFTA features',
        yaxis=dict(zeroline=False),xaxis=dict(zeroline=False)
    )
    Pfig = go.Figure(data=PData, layout=layout)
    plotly.offline.plot(Pfig,filename='SFTA-tSNE.html')
    return fig

#用于SFTA特征提取的数据加载函数
def load(filenames,nt):
    data=[]
    label=[]
    k=0
    for filename in filenames:
        for i in range(len(filename)):
            tempData=cv2.imread(filename[i])
            tempData,_=fE.segment(tempData)
            tempData=cv2.resize(tempData,dsize=(330,250))
            tempData=sfta.sfta(tempData,nt)
            print(i,'th image sfta completed!')
            data.append(tempData)
            label.append(k)
        k=k+1
    return np.array(data,dtype=np.float),np.array(label)

#用于SIFT特征提取的数据加载函数
def loadSIFT(filenames):
    data=[]
    label=[]
    k=0
    for filename in filenames:
        for i in range(len(filename)):
            tempData=cv2.imread(filename[i])
            tempData=fE.segment(tempData)
            tempData=cv2.cvtColor(tempData,cv2.COLOR_BGR2GRAY)
            tempData=cv2.resize(tempData,dsize=(330,250))
            print(i,'th image sift completed!')
            data.append(tempData)
            label.append(k)
        k=k+1
    return np.array(data,dtype=np.uint8),np.array(label)