"""
snapshot ensembling模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from math import pi
from math import cos
import multiprocessing as mp
import numpy as np

cuda=torch.cuda.is_available()

#计算学习率
def snapEnsemble_lr(initial_lr,iteration,epoch_per_cycle):
    return initial_lr*(cos(pi*iteration/epoch_per_cycle)+1)/2

#训练过程
def train(model,epochs,cycles,initial_lr,train_loader,writer):
    snapshots=[]
    _lr_list,_loss_list=[],[]
    count=0
    epochs_per_cycle=epochs//cycles
    optimizer=optim.SGD(model.parameters(),lr=initial_lr)
    totalStep=0
    loss_func = torch.nn.CrossEntropyLoss()

    for i in range(cycles):
        for j in range(epochs_per_cycle):
            _epoch_loss=0
            lr=snapEnsemble_lr(initial_lr,j,epochs_per_cycle)

            optimizer.state_dict()["param_groups"][0]["lr"]=lr
            writer.add_scalar('Learning',lr,i*epochs_per_cycle+j)
            for batch_idx,(data,target) in enumerate(train_loader):
                data=torch.tensor(data,dtype=torch.float32).to('cuda')
                target=Variable(target).type(torch.LongTensor).to('cuda')

                optimizer.zero_grad()
                output=model(data)
                loss=loss_func(output,target)
                _epoch_loss+=loss.item()/len(train_loader)
                loss.backward()
                optimizer.step()
                writer.add_scalar('Train',loss,totalStep)
                totalStep+=1
                #print(i,j,batch_idx,loss.item())
            print(i,j,loss.item())

            _lr_list.append(lr)
            _loss_list.append(_epoch_loss)
            count+=1
        snapshots.append(model.state_dict())
    return snapshots

#测试函数
def test(Model,weights,use_model_num,test_loader):
    index=len(weights)-use_model_num
    weights=weights[index:]
    model_list=[Model() for _ in weights]

    for model,weight in zip(model_list,weights):
        model.load_state_dict(weight)
        model.eval()
        model.cuda()

    test_loss=0
    correct=0
    for data,target in test_loader:
        data=torch.tensor(data,dtype=torch.float32).to('cuda')
        target=torch.tensor(target,dtype=torch.int).to('cuda')
        targetL=Variable(target).type(torch.LongTensor).to('cuda')
        output_list=[model(data).unsqueeze(0) for model in model_list]
        output=torch.mean(torch.cat(output_list),0).squeeze()
        test_loss+=F.nll_loss(output,targetL).item()
        pred=output.data.max(1)[1]
        correct+=pred.eq(target.data).cpu().sum()

    test_loss/=len(test_loader)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),100*correct/len(test_loader.dataset)
    ))