"""
用于ELM训练的伪逆求解模块
"""
import torch
from torch.autograd import Variable

#定义伪逆求解模型
class pseudoInv(object):
    def __init__(self,params,C=1e-2,forgettingfactor=1,L=100):
        self.params=list(params)
        self.C=C
        self.L=L
        self.w=self.params[len(self.params)-1]
        self.w.data.fill_(0)
        self.dimInput=self.params[len(self.params)-1].data.size()[1]
        self.forgettingfactor=forgettingfactor
        self.M=Variable(torch.inverse(self.C*torch.eye(self.dimInput)),requires_grad=False,volatile=True)
    #求解器初始化
    def initialize(self):
        self.M=Variable(torch.inverse(self.C*torch.eye(self.dimInput)),requires_grad=False,volatile=True)
        self.M=self.M.cuda()
        self.w=self.params[len(self.params)-1]
        self.w.data.fill_(0.0)
    #输入矩阵宽大于长的情形
    def pseudoBig(self,inputs,oneHotTarget):
        xtx=torch.mm(inputs.t(),inputs)
        dimInput=inputs.size()[1]
        I=Variable(torch.eye(dimInput),requires_grad=False,volatile=True)
        I=I.cuda()
        if self.L>0.0:
            mu=torch.mean(inputs,dim=0,keepdim=True)
            S=inputs-mu
            S=torch.mm(S.t(),S)
            self.M=Variable(torch.inverse(xtx.data+self.C*(I.data+self.L*S.data)),requires_grad=False,volatile=True)
        else:
            self.M=Variable(torch.inverse(xtx.data+self.C*I.data),requires_grad=False,volatile=True)

        w=torch.mm(self.M,inputs.t())
        w=torch.mm(w,oneHotTarget)
        self.w.data=w.t().data
    #输入矩阵长大于宽的情形
    def pseudoSmall(self,inputs,oneHotTarget):
        xxt=torch.mm(inputs,inputs.t())
        numSamples=inputs.size()[0]
        I=Variable(torch.eye(numSamples),requires_grad=False,volatile=True)
        I=I.cuda()
        self.M=Variable(torch.inverse(xxt.data+self.C*I.data),requires_grad=False,volatile=True)
        w=torch.mm(inputs.t(),self.M)
        w=torch.mm(w,oneHotTarget)
        self.w.data=w.t().data

    #迭代求解过程
    def train(self,inputs,targets,oneHotVectorize=True):
        targets=targets.view(targets.size(0),-1)
        if oneHotVectorize:
            targets=self.oneHotVectorize(targets=targets)
        numSamples=inputs.size()[0]
        dimInput=inputs.size()[1]
        dimTarget=targets.size()[1]
        if numSamples>dimInput:
            self.pseudoBig(inputs,targets)
        else:
            self.pseudoSmall(inputs,targets)
    #将标签进行one-hot编码
    def oneHotVectorize(self,targets):
        oneHotTarget=torch.zeros(targets.size()[0],targets.max().item()+1)
        for i in range(targets.size()[0]):
            oneHotTarget[i][targets[i].data[0]]=1
        oneHotTarget=oneHotTarget.cuda()
        oneHotTarget=Variable(oneHotTarget,requires_grad=False,volatile=True)
        return oneHotTarget