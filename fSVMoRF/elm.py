"""
极限学习机模块
"""
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from pseudoInv import pseudoInv
import matplotlib.pyplot as plt

#读取已经提取好的特征
data=np.loadtxt('data/data10.txt')
label=np.loadtxt('data/label10.txt').astype(np.int)
print(data.shape)
#定义网络的结构和相关超参数
BATCH_SIZE=20
TEST_BATCH_SIZE=20
HIDDEN_SIZE=1000
INPUT_DIM=data.shape[1]
ACTIVATION='leaky_relu'
SEED=1
#分割数据集为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.5,random_state=42,stratify=label)
#将数据转为torch数据
X_train=torch.from_numpy(X_train).float()
X_test=torch.from_numpy(X_test).float()
y_train=torch.from_numpy(y_train).int()
y_test=torch.from_numpy(y_test).int()
train_dataset=Data.TensorDataset(X_train,y_train)
train_loader=Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataset=Data.TensorDataset(X_test,y_test)
test_loader=Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)

torch.manual_seed(SEED)

#定义ELM的网络模型
class Net(nn.Module):
    def __init__(self,input_dim,hidden_size=7000,activation='leaky_relu'):
        super(Net,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_size)
        self.activation=getattr(F,activation)
        if activation in ['relu','leaky_relu']:
            torch.nn.init.xavier_uniform(self.fc1.weight,gain=nn.init.calculate_gain(activation))
        else:
            torch.nn.init.xavier_uniform(self.fc1.weight,gain=1)
        self.fc2=nn.Linear(hidden_size,6,bias=False)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.activation(x)
        x=self.fc2(x)
        return x

    def forwardToHidden(self,x):
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.activation(x)
        return x

#ELM的训练函数
def train(model,optimizer,train_loader):
    model.train()
    correct=0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.cuda(),target.cuda()
        data.target=Variable(data,requires_grad=False,volatile=True),Variable(target,requires_grad=False,volatile=True)
        hiddenOut=model.forwardToHidden(data)
        optimizer.train(inputs=hiddenOut,targets=target)
        output=model.forward(data)
        pred=output.data.max(1)[1]
        correct+=pred.eq(target.data).cpu().sum()
    print('\n Train set accuracy: {}/{} ({:.2f}%)\n'.format(correct,len(train_loader.dataset),100.*correct/len(train_loader.dataset)))

#ELM的测试函数
def mtest(model,test_loader):
    model.train()
    correct=0
    for data,target in test_loader:
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,requires_grad=False,volatile=True),Variable(target,requires_grad=False,volatile=True)
        output=model.forward(data)
        pred=output.data.max(1)[1]
        correct+=pred.eq(target.data).cpu().sum()
    print('\nTest set accuracy: {}/{} ({:.2f}%)\n'.format(correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
    return 100.*correct/len(test_loader.dataset)

#测试模块
if __name__=="__main__":
    hidden_size=np.linspace(1,1000,num=1000,dtype=int)
    print(hidden_size)
    Accs=np.zeros(1000)
    for i in range(1000):
        print('hidden_size:',hidden_size[i])
        model = Net(input_dim=INPUT_DIM, hidden_size=hidden_size[i], activation=ACTIVATION)
        model.cuda()
        optimizer = pseudoInv(params=model.parameters(), C=0.001, L=0)
        train(model,optimizer,train_loader)
        Accs[i]=mtest(model,test_loader)
    x=np.arange(1000)
    plt.plot(x,Accs)
    plt.show()