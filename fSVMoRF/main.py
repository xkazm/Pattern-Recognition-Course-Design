"""
方案三和方案四：基于Sparse-SIFT和STFA的花卉识别方法
"""
import numpy as np
import glob
import load
import cv2
import featureExtract as fE
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import sfta

#matplotlib.use('TkAgg')
#读取图片文件
filenames=[]
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/buttercup/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/daisy/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/iris/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/lilyvalley/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/sunflower/*.jpg'))
filenames.append(glob.glob(r'F:/seven/prDesign/fdata/windflower/*.jpg'))

nt=10#设定阈值分割的阈值个数

#加载图片文件并做STFA特征提取，将提取出的特征存入对应文件中
#data,label=load.load(filenames,nt)
#np.savetxt('F:/seven/prDesign/fSVMoRF/data/data.txt',data)
#np.savetxt('F:/seven/prDesign/fSVMoRF/data/label.txt',label)

#读取已经获得的STFA特征或者Sparse-SIFT特征
data=np.loadtxt('data/nx.txt')
label=np.loadtxt('data/ny.txt').astype(np.int)
print(data.shape)
dmin=np.min(data,axis=0)
dmax=np.max(data,axis=0)
#data=(data-dmin)/(dmax-dmin)#归一化
#print(data)
#print(label)

#提取Sparse-SIFT特征并存入对应文件
#data,label=load.loadSIFT(filenames)
#print(data.shape)
#sdata=fE.SIFT(data)
#np.savetxt('data/sift.txt',sdata)
#print(sdata.shape)

#数据可视化
tsne=manifold.TSNE(n_components=2,init='pca',random_state=0)
Y=tsne.fit_transform(data)
print('tSNE completed!')
fig=load.plot_embedding(Y,label,'t-SNE embedding of the SIFT features')
plt.show()

#分割数据集为训练集和测试集
train_X,test_X,train_y,test_y=sklearn.model_selection.train_test_split(data,label,test_size=0.5,random_state=42,stratify=label)
#读取测试图片的特征
#testFile='F:/seven/prDesign/test6.jpg'
#tData=cv2.imread(testFile)
#tData,_=fE.segment(tData)
#ttData=[]
#tData=cv2.cvtColor(tData,cv2.COLOR_BGR2GRAY)
#tData=cv2.resize(tData,dsize=(330,250))
#ttData.append(tData)
#ttData=np.array(ttData)
#tData=sfta.sfta(tData,nt)
#print('image sfta completed!')
#tData=fE.SIFT(ttData)
#tData=(tData-dmin)/(dmax-dmin)#归一化
#决策树分类模型
clf1=DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
clf1.fit(train_X,train_y)
score1=clf1.score(test_X,test_y)
print('Decision Tree:',score1)
#print(clf1.predict(tData.reshape(1,-1)))
np.set_printoptions(precision=2)
name=['buttercup','daisy','iris','lilyvalley','sunflower','windflower']
# Plot non-normalized confusion matrix
titles_options = [("Normalized confusion matrix for Decision Tree", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf1, test_X, test_y,
                                 display_labels=name,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
#随机森林分类模型
clf2=RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
clf2.fit(train_X,train_y)
score2=clf2.score(test_X,test_y)
print('Random Forest:',score2)
#print(clf2.predict(tData.reshape(1,-1)))
titles_options = [("Normalized confusion matrix for Random Forest", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf2, test_X, test_y,
                                 display_labels=name,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
#支持向量机模型
clf3=sklearn.svm.SVC()
clf3.fit(train_X,train_y)
score3=clf3.score(test_X,test_y)
print('SVM:',score3)
#print(clf3.predict(tData.reshape(1,-1)))
titles_options = [("Normalized confusion matrix for SVM", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf3, test_X, test_y,
                                 display_labels=name,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
