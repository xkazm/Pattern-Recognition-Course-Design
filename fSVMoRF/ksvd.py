"""
K-SVD稀疏编码模块
"""
import numpy as np
from sklearn import linear_model

#定义K-SVD稀疏编码方法
class KSVD(object):
    def __init__(self,n_components,max_iter=10,tol=1e-6):
        self.dictionary=None
        self.sparsecode=None
        self.max_iter=max_iter
        self.tol=tol
        self.n_components=n_components

    def _initialize(self,y):
        u,s,v=np.linalg.svd(y)
        self.dictionary=u[:,:self.n_components]

    def _update_dict(self,y,d,x):
        for i in range(self.n_components):
            index=np.nonzero(x[i,:])[0]
            if len(index)==0:
                continue

            d[:,i]=0
            r=(y-np.dot(d,x))[:,index]
            u,s,v=np.linalg.svd(r,full_matrices=False)
            x[i,index]=s[0]*v[0,:]
        return d,x

    def fit(self,y):
        self._initialize(y)
        for i in range(self.max_iter):
            x=linear_model.orthogonal_mp(self.dictionary,y)
            print('linear model')
            e=np.linalg.norm(y-np.dot(self.dictionary,x))
            if e<self.tol:
                self._update_dict(y,self.dictionary,x)
            print(i,'th ksvd completed!')

        self.sparsecode=linear_model.orthogonal_mp(self.dictionary,y)
        return self.dictionary,self.sparsecode