from skimage import filters
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pickle as pkl
from itertools import product,chain
from scipy.misc import face
from scipy.ndimage import convolve
import cv2
from sklearn import cluster
from sklearn import mixture
import os,sys
import argparse
from sklearn.externals import joblib

class MR8_FilterBank():
    def __init__(self,sigma=[1,2,4],n_orienatation=6):
        return

    def make_gaussian_filter(self,x,sigma,order=0):
        if order>2:
            raise ValueError("Only orders up to 2 are supported")
        response=np.exp(-x**2/(2. *sigma**2))
        if order==1:
            response=-response*x
        elif order==2:
            response=response*(x**2-sigma**2)
        response/=np.abs(response).sum()
        return response

    def makefilter(self,scale,phasey,pts,sup):
        gx=self.make_gaussian_filter(pts[0,:],sigma=3*scale)
        gy=self.make_gaussian_filter(pts[1,:],sigma=scale,order=phasey)
        f=(gx*gy).reshape(sup,sup)
        f/=np.abs(f).sum()
        return f

    def makeRFSfilters(self,radius=11,sigmas=[1,2,4],n_orientations=6):
        support=2*radius+1
        x,y=np.mgrid[-radius:radius+1,radius:-radius-1:-1]
        orgpts=np.vstack([x.ravel(),y.ravel()])

        rot,edge,bar=[],[],[]
        for sigma in sigmas:
            for orient in range(n_orientations):
                angle=np.pi*orient/n_orientations
                c,s=np.cos(angle),np.sin(angle)
                rotpts=np.dot(np.array([[c,-s],[s,c]]),orgpts)
                edge.append(self.makefilter(sigma,1,rotpts,support))
                bar.append(self.makefilter(sigma,2,rotpts,support))
        length=np.sqrt(x**2+y**2)
        rot.append(self.make_gaussian_filter(length,sigma=10))
        rot.append(self.make_gaussian_filter(length,sigma=10,order=2))

        edge=np.asarray(edge)
        edge=edge.reshape((len(sigmas),n_orientations,support,support))
        bar=np.asarray(bar).reshape(edge.shape)
        rot=np.asarray(rot)[:,np.newaxis,:,:]
        return edge,bar,rot

    def apply_filterbank(self,img,filterbank):
        result=[]
        for battery in filterbank:
            response=[convolve(img,filt,mode='reflect') for filt in battery]
            max_response=np.max(response,axis=0)
            result.append(max_response)
            #print("battery finished")
        return result

"""
if __name__=="__main__":
    sigmas=[1,2,4]
    n_sigmas=len(sigmas)
    n_orientations=6
    mr8=MR8_FilterBank()
    edge,bar,rot=mr8.makeRFSfilters(sigmas=sigmas,n_orientations=n_orientations)

    n=n_sigmas*n_orientations

    fig,ax=plt.subplots(n_sigmas*2+1,n_orientations)
    for k,filters in enumerate([bar,edge]):
        for i,j in product(range(n_sigmas),range(n_orientations)):
            row=i+k*n_sigmas
            ax[row,j].imshow(filters[i,j,:,:],cmap=plt.cm.gray)
            ax[row,j].set_xticks(())
            ax[row,j].set_yticks(())
    ax[-1,0].imshow(rot[0,0],cmap=plt.cm.gray)
    ax[-1,0].set_xticks(())
    ax[-1,0].set_yticks(())
    ax[-1,1].imshow(rot[1,0],cmap=plt.cm.gray)
    ax[-1,1].set_xticks(())
    ax[-1,1].set_yticks(())
    for i in range(2,n_orientations):
        ax[-1,i].set_visible(False)

    img=face(gray=True).astype(np.float)
    print(img.shape)
    filterbank=chain(edge,bar,rot)
    n_filters=len(edge)+len(bar)+len(rot)
    print('[applying]:%d'%n_filters)
    response=mr8.apply_filterbank(img,filterbank)

    fig2,ax2=plt.subplots(3,3)
    for axes,res in zip(ax2.ravel(),response):
        axes.imshow(res,cmap=plt.cm.gray)
        axes.set_xticks(())
        axes.set_yticks(())
    ax2[-1,-1].set_visible(False)
    plt.show()
"""

class MR8_Vector():
    def __init__(self,save_flag=False,out_dir='.'):
        self.outdir=out_dir

    def create_mr8_features(self,img,sigmas=[1,2,4],n_ort=6):
        #if img==None:
            #raise Exception('No image found')
        self.mr8bank=MR8_FilterBank()
        self.edge,self.bar,self.rot=self.mr8bank.makeRFSfilters(sigmas=sigmas,n_orientations=n_ort)
        self.filterbank=chain(self.edge,self.bar,self.rot)
        self.n_filters=len(self.edge)+len(self.bar)+len(self.rot)
        self.filterbank=chain(self.edge,self.bar,self.rot)
        self.responses=self.mr8bank.apply_filterbank(img,self.filterbank)
        return self.responses

    def create_feature_vectors(self,img):
        #if img==None:
            #raise Exception('No image found')
        responses=self.create_mr8_features(img)
        out_img=np.zeros((img.shape[0],img.shape[1],8),dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                vec=np.zeros((8,),dtype=np.float32)
                for r in range(len(responses)):
                    vec[r]=responses[r][i,j]
                l2_norm=np.linalg.norm(vec)
                if l2_norm==0:
                    l2_norm=1

                    vec=vec*(np.log(1+l2_norm)/0.03)/l2_norm
                    new_vec=np.zeros((8,),dtype=np.float32)
                    new_vec[:8]=vec[:]
                    #cent_dist=np.linalg.norm(np.asarray([i,j])-np.asarray([128.,128.]))/128.
                    #angle=np.arctan(abs(i-128)/abs(j-128)) if j-128!=0 else 0
                    #cent_flag=np.sign(j-128)
                    out_img[i,j,:]=new_vec[:]
        return out_img

    def create_vocabulary(self,features,num_clusters=700):
        self.gmm=mixture.GaussianMixture(num_clusters)
        self.gmm.fit(features)
        print('GMM Created.')
        return self.gmm
    
    def create_fv(self,ftr,gmm,alpha=0.5):
        self.gmm=gmm
        means=self.gmm.means_
        covar=self.gmm.covariances_
        weights=self.gmm.weights_
        n_comps=self.gmm.n_components
        out=np.zeros((n_comps,2*ftr.shape[1]),dtype=np.float32)
        probs=self.gmm.predict_proba(ftr)
        for k in range(n_comps):
            sum_k1=0
            sum_k2=0
            T=ftr.shape[0]
            check_flag=0
            for i in range(ftr.shape[0]):
                if probs[i,k]<0.001:
                    check_flag+=1
                    continue
                sum_k1+=((ftr[i,:]-means[k])/covar[k])*probs[i,k]
                sum_k2+=((((ftr[i,:]-means[k])**2)/covar[k]**2)-1)*probs[i,k]
            if check_flag==ftr.shape[0]:
                sum_k1=np.zeros(ftr[0].shape,dtype=np.float32)
                sum_k2=np.zeros(ftr[0].shape,dtype=np.float32)
            sum_k1=(1/(T*np.sqrt(weights[k]+0.00001)))*sum_k1
            sum_k2=(1/(T*np.sqrt(2*weights[k]+0.00001)))*sum_k2
            print(sum_k1.shape)
            print(sum_k2.shape)
            sum_out=np.concatenate((sum_k1,sum_k2),axis=0)
            print(sum_out.shape)
            sum_out=np.sign(sum_out)*np.abs(sum_out)**alpha
            sum_out=sum_out/(np.linalg.norm(sum_out)+0.00001)
            out[k,:]=sum_out
        print('Out',out.shape)
        return out.flatten()

if __name__=="__main__":
    fv=MR8_Vector()
    loc_dict=dict()
    outpath="F:/seven/prDesign/test/"
    filename='F:/seven/prDesign/fdata/windflower/image_1268.jpg'
    img=cv2.imread("F:/seven/prDesign/fdata/windflower/image_1268.jpg",cv2.IMREAD_COLOR)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('Creating Features for %s'%filename)
    ftr=[]
    temp=fv.create_feature_vectors(img)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            ftr.append(temp[i,j,:])
    local_descriptors=np.asarray(ftr)
    loc_dict[filename]=local_descriptors
    joblib.dump(loc_dict,outpath+'_feature_dict.pkl')
    print('Creating Feature end')
    img_dict=joblib.load(outpath+'_feature_dict.pkl')
    ftrs=[]
    for img_name in img_dict.keys():
        ftrs.append(img_dict[img_name])
    ftrs_in=np.asarray(ftrs)[0,:,:]
    vocab=fv.create_vocabulary(ftrs_in,700)
    print('Creating Vocabulary end')
    fvectors=dict()
    fvectors[filename]=fv.create_fv(ftr=local_descriptors,gmm=vocab)
    print(fvectors)