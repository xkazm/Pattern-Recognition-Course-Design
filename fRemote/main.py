"""
神经网络模块——方案五、六和七
"""
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,models,transforms
from math import pi
from math import cos
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms,utils
import torchvision
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
import numpy as np
import glob
import load
from sklearn.model_selection import train_test_split
import scipy.io as scio
import cv2
import snapensemble
#import FCN
import os
from sklearn import manifold,datasets
import matplotlib.pyplot as plt

filenames=[]
#filenames=sorted(glob.glob(r'oxford102/segmim/*.jpg'))
testFile='test6.jpg'
print(filenames)
#"""6类问题数据集加载
filenames.append(glob.glob(r'fdata/buttercup/*.jpg'))
filenames.append(glob.glob(r'fdata/daisy/*.jpg'))
filenames.append(glob.glob(r'fdata/iris/*.jpg'))
filenames.append(glob.glob(r'fdata/lilyvalley/*.jpg'))
filenames.append(glob.glob(r'fdata/sunflower/*.jpg'))
filenames.append(glob.glob(r'fdata/windflower/*.jpg'))
#"""
data,label=load.load(filenames)
#label=tf.one_hot(label,6)
print(data.shape)
"""
data=[]
for filename in filenames:
    print(filename)
    tempData=cv2.imread(filename,cv2.IMREAD_COLOR)
    tempData=cv2.cvtColor(tempData,cv2.COLOR_BGR2HSV)
    tempData=cv2.resize(tempData,(299,299))
    tempData=np.swapaxes(tempData,0,2)
    tempData=tempData/255.
    data.append(tempData)
data=np.array(data,dtype=float)

tData=[]
testData=cv2.imread(testFile,cv2.IMREAD_COLOR)
testData=load.segmentation(testData)
testData=cv2.cvtColor(testData,cv2.COLOR_BGR2HSV)
testData=cv2.resize(testData,(100,100))
testData=np.swapaxes(testData,0,2)
testData=testData/255.
tData.append(testData)
tData=np.array(tData)

y=scio.loadmat('oxford102/imagelabels.mat')
label=np.array(y['labels']).reshape(-1)-1
print(y)
"""
#writer=SummaryWriter()

BATCH_SIZE=40
CLASS_NUM=6
EPOCH=50
LR=0.0001 #for 6 classes
#LR=0.000005

X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.5,random_state=42,stratify=label)
X_train=torch.from_numpy(X_train).double()
X_test=torch.from_numpy(X_test).double()
y_train=torch.from_numpy(y_train).int()
y_test=torch.from_numpy(y_test).int()
data_v=torch.from_numpy(data).double()
data_l=torch.from_numpy(label).int()
#TESTX=torch.from_numpy(tData).double()
print('nu1')
train_dataset=Data.TensorDataset(X_train,y_train)
train_loader=Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataset=Data.TensorDataset(X_test,y_test)
test_loader=Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

VisualData=Data.TensorDataset(data_v,data_l)
VisualData_loader=Data.DataLoader(dataset=VisualData,batch_size=BATCH_SIZE,shuffle=True)
print('nu2')

def mforward(model, x):
    if model.transform_input:
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    # N x 3 x 299 x 299
    x = model.Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x = model.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = model.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # N x 64 x 73 x 73
    x = model.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = model.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # N x 192 x 35 x 35
    x = model.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = model.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = model.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = model.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6e(x)
    # N x 768 x 17 x 17
    if model.training and model.aux_logits:
        aux = model.AuxLogits(x)
    # N x 768 x 17 x 17
    x = model.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = model.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = model.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = F.adaptive_avg_pool2d(x, (1, 1))
    # N x 2048 x 1 x 1
    x = F.dropout(x, training=model.training)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    return x

#自定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.fc1=nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU()
        )
        self.out=nn.Linear(512,6)

        # 迭代循环初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,1.)
            # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        #x=torch.nn.functional.dropout(x,p=0.5)
        x=self.fc2(x)
        #x=torch.nn.functional.dropout(x,p=0.5)
        output=self.out(x)
        output=torch.nn.functional.relu(output)
        return output

#使用自定义卷积神经网络模型
#cnn=CNN().to("cuda")
#simInput=torch.randn(40,3,100,100).to('cuda')

#optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
#optimizer=torch.optim.SGD(cnn.parameters(),lr=LR,momentum=0.9)
#loss_function=nn.CrossEntropyLoss()
print('nu3')
totalStep=0
#fcn_model=torch.load('fcn_model_1000').cuda()
#使用预训练模型
model_conv=torchvision.models.inception_v3(pretrained=True)
print('nu4')
#model_conv=torch.load("model_conv")
#model_conv=torchvision.models.resnet101(pretrained=True)
#model_conv=torchvision.models.vgg16(pretrained=True)
#固定预训练模型的参数（除最后一层），不参与训练
for param in model_conv.parameters():
    param.requires_grad=False

print('have executed')
#重新定义最后一层网络结构，并用于训练
num_ftrs=model_conv.fc.in_features
model_conv.fc=nn.Linear(num_ftrs,6)
#model_conv.classifier._modules['6']=nn.Linear(4096,102)
model_conv.aux_logits=False

#params = [{'params': md.parameters()} for md in model_conv.children()
          #if md in [model_conv.classifier]]

#model_conv=model_conv.to('cuda')
#simInput=torch.randn(20,3,299,299).to('cuda')

loss_function=nn.CrossEntropyLoss()
#optimizer_conv=optim.SGD(model_conv.parameters(),lr=0.01,momentum=0.9)
optimizer_conv=optim.SGD(model_conv.parameters(),lr=0.01,momentum=0.9)

#with SummaryWriter(comment='Net1') as w:
    #w.add_graph(model_conv,[simInput,])

#可视化
print('enter Visualization')
visualX=np.zeros((480,2048))
visualY=np.zeros(480)
k=0
for vdata in VisualData_loader:
    print(k)
    x,y=vdata
    v_x=torch.tensor(x,dtype=torch.float32)
    v_y=torch.tensor(y,dtype=torch.int)
    output=mforward(model_conv,v_x)
    output_numpy=output.numpy()
    Y_numpy=v_y.numpy()
    visualX[k*40:(k+1)*40,:]=output_numpy
    visualY[k*40:(k+1)*40]=Y_numpy
    k=k+1
visualX=np.array(visualX)
visualY=np.array(visualY,dtype=int)
print(visualX.shape)
print(visualY.shape)
np.savetxt('nx.txt',visualX)
np.savetxt('ny.txt',visualY)

tsne=manifold.TSNE(n_components=2,init='pca',random_state=0)
Y=tsne.fit_transform(visualX)
print('tSNE completed!')
fig=load.plot_embedding(Y,visualY,'t-SNE embedding of the feature from Inception-V3 model')
plt.show()

"""
#常规训练过程训练
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=torch.tensor(x,dtype=torch.float32).to('cuda')
        b_y=Variable(y).type(torch.LongTensor).to('cuda')
        optimizer_conv.zero_grad()

        output=model_conv(b_x)
        #print(output,b_y)
        loss=loss_function(output,b_y)
        loss.backward()
        optimizer_conv.step()

        if step%100==0:
            #accuracy=0
            sum=0
            for tdata in test_loader:
                x,y=tdata
                t_x=torch.tensor(x,dtype=torch.float32).to('cuda')
                t_y=torch.tensor(y,dtype=torch.int).to('cuda')
                test_output=model_conv(t_x)
                pred_y=torch.max(test_output,1)[1].data.squeeze()
                for k in range(t_y.size(0)):
                    if pred_y[k]==t_y[k]:
                        sum=sum+1
            accuracy=sum/y_test.size(0)
            print('Epoch:',epoch,"|Step:",step,"|train loss:%.4f"%loss.item(),"|test accuracy:%.4f"%accuracy)
            #writer.add_scalar('Test',accuracy,totalStep)
        #writer.add_scalar('Train',loss,totalStep)
        totalStep=totalStep+1
"""
#预测输出结果示
#TestT=torch.tensor(TESTX,dtype=torch.float32).to('cuda')
#test_output=model_conv(TestT)
#pred_y=torch.max(test_output,1)[1].data.squeeze()
#print(pred_y,'prediction number')
#print(y_test[:10],'real number')
model_conv=model_conv.to('cpu')
torch.save(model_conv,'model_conv')
#snapshot ensemble训练过程
#models=snapensemble.train(cnn,300,6,0.01,train_loader,writer)
#snapensemble.test(CNN,models,5,test_loader)

#writer.close()
