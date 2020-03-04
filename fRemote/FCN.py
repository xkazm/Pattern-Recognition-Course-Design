import cv2
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import models
from torchvision.models.vgg import VGG
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
import pdb
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#matplotlib.use('Qt5Agg')

def onehot(data,n):
    buf=np.zeros(data.shape+(n,))
    nmsk=np.arange(data.size)*n+data.ravel()
    buf.ravel()[nmsk-1]=1
    return buf

class BagDataset(Dataset):
    def __init__(self,filenames,msk_filenames):
        self.filenames=filenames
        self.msk_filenames=msk_filenames
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self,idx):
        img_name=self.filenames[idx]
        msk_img_name=self.msk_filenames[idx]
        img=cv2.imread(img_name)
        img=cv2.resize(img,(160,160))
        img_msk=cv2.imread(msk_img_name)
        #img_msk = cv2.cvtColor(img_msk, cv2.COLOR_BGR2GRAY)
        img_msk=cv2.resize(img_msk,(160,160))
        img_msk=img_msk/255
        img_msk=img_msk.astype('uint8')
        img_msk=onehot(img_msk,2)
        img=img.swapaxes(0,2).swapaxes(1,2)
        img_msk=img_msk.swapaxes(0,2).swapaxes(1,2)
        img=torch.FloatTensor(img)
        img_msk=torch.FloatTensor(img_msk)
        item={'ori':img,'msk':img_msk}
        return item


class FCN(nn.Module):
    def __init__(self,pretrained_net,n_class):
        super().__init__()
        self.n_class=n_class
        self.pretrained_net=pretrained_net
        self.relu=nn.ReLU(inplace=True)
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn4=nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,dilation=1,output_padding=1)
        self.bn5=nn.BatchNorm2d(32)
        self.classifier=nn.Conv2d(32,n_class,kernel_size=1)

    def forward(self,x):
        output=self.pretrained_net(x)
        x5=output['x5']
        x4=output['x4']
        x3=output['x3']

        score=self.relu(self.deconv1(x5))
        score=self.bn1(score+x4)
        score=self.relu(self.deconv2(score))
        score=self.bn2(score+x3)
        score=self.bn3(self.relu(self.deconv3(score)))
        score=self.bn4(self.relu(self.deconv4(score)))
        score=self.bn5(self.relu(self.deconv5(score)))
        score=self.classifier(score)

        return score

ranges={
    'vgg11':((0,3),(3,6),(6,11),(11,16),(16,21)),
    'vgg13':((0,5),(5,10),(10,15),(15,20),(20,25)),
    'vgg16':((0,5),(5,10),(10,17),(17,24),(24,31)),
    'vgg19':((0,5),(5,10),(10,19),(19,28),(28,37))
}

cfg={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

def make_layers(cfg,batch_norm=False):
    layers=[]
    in_channels=3
    for v in cfg:
        if v=='M':
            layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers+=[conv2d,nn.ReLU(inplace=True)]
            in_channels=v
    return nn.Sequential(*layers)

class VGGNet(VGG):
    def __init__(self,pretrained=True,model='vgg16',requires_grad=False,remove_fc=True):
        super().__init__(make_layers(cfg[model]))
        self.ranges=ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())"%model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad=False

        if remove_fc:
            del self.classifier

    def forward(self,x):
        output={}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0],self.ranges[idx][1]):
                x=self.features[layer](x)
            output["x%d"%(idx+1)]=x

        return output

if __name__=="__main__":
    #filenames = sorted(glob.glob(r'oxford102/jpg/*.jpg'))
    filenames=sorted(glob.glob(r'F:/seven/prDesign/test*.jpg'))
    #msk_filenames=sorted(glob.glob(r'oxford102/mask/*.jpg'))
    BATCH_SIZE=40
    EPOCHS=1000

    #bag=BagDataset(filenames,msk_filenames)
    #dataLoader=DataLoader(bag,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    #vgg_model=VGGNet(requires_grad=True)
    #fcn_model=FCN(pretrained_net=vgg_model,n_class=2)
    #fcn_model=fcn_model.cuda()
    fcn_model=torch.load('fcn_model_1000').cuda()
    #loss_fn=nn.BCELoss().cuda()
    #optimizer=optim.SGD(fcn_model.parameters(),lr=0.01,momentum=0.9)

    saving_index=0
    """
    for epoch in range(EPOCHS):
        saving_index+=1
        index=0
        epoch_loss=0
        for item in dataLoader:
            index+=1
            x=item['ori']
            y=item['msk']

            x=torch.autograd.Variable(x).to('cuda')
            y=torch.autograd.Variable(y).to('cuda')

            optimizer.zero_grad()
            output=fcn_model(x)
            output=nn.functional.sigmoid(output)
            loss=loss_fn(output,y)
            
            loss.backward()
            iter_loss=loss.item()
            epoch_loss+=iter_loss
            optimizer.step()
            
            #output_np=output.cpu().data.numpy().copy()
            #output_np=np.argmin(output_np,axis=1)
            #y_np=y.cpu().data.numpy().copy()
            #y_np=np.argmin(y_np,axis=1)
            
            #plt.subplot(1,2,1)
            #plt.imshow(np.squeeze(y_np[0,:,:]),'gray')
            #plt.subplot(1,2,2)
            #plt.imshow(np.squeeze(output_np[0,:,:]),'gray')
            #plt.savefig(str(index+1)+'.png')
            #plt.show()
            #plt.pause(0.5)

        print('epoch loss = %f'%(epoch_loss/len(dataLoader)))

        if np.mod(saving_index,5)==0:
            torch.save(fcn_model,'checkpoint/fcn_model_{}'.format(epoch+1))
            print('saving checkpoints/fcn_model_{}.pt'.format(epoch+1))
    """
    for i in range(len(filenames)):
        img=cv2.imread(filenames[i])
        img=cv2.resize(img,(160,160)).astype(np.uint8)
        img_e=img.reshape(1,3,160,160)
        img_ten=torch.tensor(img_e,dtype=torch.float32).to('cuda')
        img_cu = torch.autograd.Variable(img_ten).to('cuda')
        img_mask=fcn_model(img_cu)
        print(i)
        img_mask_np=img_mask.cpu().data.numpy().copy()
        img_mask_np = np.argmin(img_mask_np, axis=1)
        img_mask_np=(img_mask_np+1/255).astype(np.uint8).reshape(160,160)
        #print(img_mask_np)
        print(img.shape,img_mask_np.shape)
        img_seg=cv2.bitwise_and(img,img,mask=img_mask_np)
        cv2.imwrite('seg'+str(i+1)+'3.jpg',img_seg,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #cv2.imshow('seg',img_seg)
        #cv2.waitKey(0)
