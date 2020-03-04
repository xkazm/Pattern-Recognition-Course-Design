"""
特征提取模块
"""
import numpy as np
import cv2
import dSIFT
from sklearn.cluster import MiniBatchKMeans

#矩阵乘方
def matrixPow(X,gamma):
    m,n=X.shape
    v,Q=np.linalg.eig(X)
    V=np.diag(v**(gamma))
    result=np.dot(Q,V)
    result=np.dot(result,np.mat(Q).T)
    return np.real(result)
#按空间金字塔要求将图像4等分
def splitImage(img):
    m,n=img.shape[0],img.shape[1]
    centerR=int(m/2)
    centerC=int(n/2)
    img1=img[:centerR,:centerC,:]
    img2=img[:centerR,centerC:,:]
    img3=img[centerR:,:centerC,:]
    img4=img[centerR:,centerC:,:]
    return img1,img2,img3,img4
#按空间金字塔要求将图像4等分
def splitGEImage(img):
    m, n = img.shape[0], img.shape[1]
    centerR = int(m / 2)
    centerC = int(n / 2)
    img1 = img[:centerR, :centerC]
    img2 = img[:centerR, centerC:]
    img3 = img[centerR:, :centerC]
    img4 = img[centerR:, centerC:]
    return img1, img2, img3, img4
#图像分割
def segmentation(img):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _, _, imgV = cv2.split(imgHSV)
    m,n,c=imgHSV.shape
    imgV=np.array(imgV)/255
    imgVF=imgV.flatten()
    meanV=np.mean(imgVF,axis=0)
    varV=np.std(imgVF,axis=0)
    Threshold=meanV+varV
    fmask=np.zeros((m,n),dtype=np.uint8)
    mask=np.array(imgV>Threshold,dtype=np.uint8)
    img_segmentation=cv2.bitwise_and(img,img,mask=mask)
    return img_segmentation
#获得编码字典
def getCodebook(H):
    kmean=MiniBatchKMeans(n_clusters=300)
    kmean.fit(H)
    return kmean.cluster_centers_
#LLC编码函数
def LLCoding(codebook,x):
    disMatrix=np.zeros(len(codebook))
    for i in range(len(codebook)):
        disMatrix[i]=np.linalg.norm(codebook[i,:]-x)
    index=np.argsort(disMatrix,axis=0)
    Chosenindex=index[:60]
    B_new=codebook[Chosenindex,:]
    e1=np.ones((60,1),dtype=float)
    bm=B_new-np.dot(e1,x.reshape(1,-1))
    b_hat=np.dot(bm,np.mat(bm).T)
    c_hat=np.dot(matrixPow(b_hat,-1.),e1).reshape(-1)
    c=c_hat/np.linalg.norm(c_hat)
    return c
#保存编码字典
def saveCodebook(pathName,codebook):
    np.savetxt(pathName,codebook)
#提取HSV编码字典和特征编码
def extractHSVCodebook(filenames,HcodeBookname,ScodeBookname,VcodeBookname,Hcodename,Scodename,Vcodename,split=False,split_po=0,level=0):
    n = len(filenames)
    print(filenames[0])
    densift = dSIFT.DenseSIFT()
    H = None
    lenH=np.zeros(len(filenames),dtype=int)
    S = None
    lenS=np.zeros(len(filenames),dtype=int)
    V = None
    lenV=np.zeros(len(filenames),dtype=int)
    for i in range(n):
        print(i)
        img = cv2.imread(filenames[i])
        img=segmentation(img)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if split:
            if level==1:
                simgHSV=splitImage(imgHSV)
                imgHSV=simgHSV[split_po]
            elif level==2:
                simgHSV=splitImage(imgHSV)
                chose_po=int(split_po/4)
                which_po=split_po%4
                imgHSV=splitImage(simgHSV[chose_po])[which_po]
        desH = densift.detectAndCompute(imgHSV[:, :, 0])
        lenH[i]=len(desH)
        desS = densift.detectAndCompute(imgHSV[:, :, 1])
        lenS[i]=len(desS)
        desV = densift.detectAndCompute(imgHSV[:, :, 2])
        lenV[i]=len(desV)
        if S is None:
            H = desH
            S = desS
            V = desV
        else:
            H = np.concatenate((H, desH), axis=0)
            S = np.concatenate((S, desS), axis=0)
            V = np.concatenate((V, desV), axis=0)
    print('dSIFT complete')
    Hcodebook = getCodebook(H)
    Scodebook = getCodebook(S)
    Vcodebook = getCodebook(V)
    saveCodebook(HcodeBookname,Hcodebook)
    saveCodebook(ScodeBookname,Scodebook)
    saveCodebook(VcodeBookname,Vcodebook)

    sumH=0
    sumS=0
    sumV=0
    HCode=np.zeros((n,60))
    SCode=np.zeros((n,60))
    VCode=np.zeros((n,60))
    for i in range(n):
        print(i,"th LLC coding")

        tempH=np.zeros((lenH[i],60))
        for k in range(lenH[i]):
            tempH[k,:]=LLCoding(Hcodebook,H[sumH+k,:])
        sumH=sumH+lenH[i]
        HCode[i,:]=np.mean(tempH,axis=0)

        tempS=np.zeros((lenS[i],60))
        for k in range(lenS[i]):
            tempS[k,:]=LLCoding(Scodebook,S[sumS+k,:])
        sumS=sumS+lenS[i]
        SCode[i,:]=np.mean(tempS,axis=0)

        tempV=np.zeros((lenV[i],60))
        for k in range(lenV[i]):
            tempV[k,:]=LLCoding(Vcodebook,V[sumV+k,:])
        sumV=sumV+lenV[i]
        VCode[i,:]=np.mean(tempV,axis=0)


    np.savetxt(Hcodename,HCode)
    np.savetxt(Scodename,SCode)
    np.savetxt(Vcodename,VCode)

#提取HSV的PHOW特征
def PHOW_HSV(img,Hcodebookname,Scodebookname,Vcodebookname,split=False,split_po=0,level=0):
    Hcodebook=np.loadtxt(Hcodebookname)
    Scodebook=np.loadtxt(Scodebookname)
    Vcodebook=np.loadtxt(Vcodebookname)

    img = segmentation(img)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if split:
        if level == 1:
            simgHSV = splitImage(imgHSV)
            imgHSV = simgHSV[split_po]
        elif level == 2:
            simgHSV = splitImage(imgHSV)
            chose_po = int(split_po / 4)
            which_po = split_po % 4
            imgHSV = splitImage(simgHSV[chose_po])[which_po]
    densift=dSIFT.DenseSIFT()
    desH=densift.detectAndCompute(imgHSV[:,:,0])
    desS=densift.detectAndCompute(imgHSV[:,:,1])
    desV=densift.detectAndCompute(imgHSV[:,:,2])
    HCode=np.zeros((len(desH),60))
    SCode=np.zeros((len(desS),60))
    VCode=np.zeros((len(desV),60))
    for i in range(len(desH)):
        #print(H[i,:].shape)
        print('H coding:',i)
        HCode[i,:]=LLCoding(Hcodebook,desH[i,:])
    for i in range(len(desS)):
        print('S coding:',i)
        SCode[i,:]=LLCoding(Scodebook,desS[i,:])
    for i in range(len(desV)):
        print('V coding:',i)
        VCode[i,:]=LLCoding(Vcodebook,desV[i,:])
    print('LLC coding complete')

    Hfinal=np.mean(HCode[:],axis=0).reshape(1,-1)
    Sfinal=np.mean(SCode[:],axis=0).reshape(1,-1)
    Vfinal=np.mean(VCode[:],axis=0).reshape(1,-1)

    return Hfinal,Sfinal,Vfinal
#存储HSV的特征编码
def saveHSVcode(filenames,Hcodebookname,Scodebookname,Vcodebookname,Hcodename,Scodename,Vcodename,split=False,split_po=0,level=0):
    Filename=[]
    for filename in filenames:
        for f in filename:
            Filename.append(f)
    extractHSVCodebook(Filename,Hcodebookname,Scodebookname,Vcodebookname,Hcodename,Scodename,Vcodename,split,split_po,level)

#提取灰度图的编码字典和特征编码
def extractGCodebook(filenames,GcodeBookname,Gcodename,split=False,level=0,split_po=0):
    n = len(filenames)
    print(filenames[0])
    densift = dSIFT.DenseSIFT()
    G = None
    lenG=np.zeros(n,dtype=int)
    for i in range(n):
        print(i)
        img = cv2.imread(filenames[i])
        img=segmentation(img)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if split:
            if level==1:
                simgHSV=splitGEImage(imgGray)
                imgGray=simgHSV[split_po]
            elif level==2:
                simgHSV=splitGEImage(imgGray)
                chose_po=int(split_po/4)
                which_po=split_po%4
                imgGray=splitGEImage(simgHSV[chose_po])[which_po]
        desG = densift.detectAndCompute(imgGray[:, :])
        lenG[i]=len(desG)
        if G is None:
            G = desG
        else:
            G = np.concatenate((G, desG), axis=0)
    print('dSIFT complete')
    Gcodebook = getCodebook(G)
    saveCodebook(GcodeBookname,Gcodebook)

    sumG = 0
    GCode = np.zeros((n, 60))
    for i in range(n):
        print(i, "th LLC coding")
        tempG = np.zeros((lenG[i], 60))
        for k in range(lenG[i]):
            tempG[k, :] = LLCoding(Gcodebook, G[sumG + k, :])
        sumG = sumG + lenG[i]
        GCode[i, :] = np.mean(tempG, axis=0)

    np.savetxt(Gcodename, GCode)
#提取edge-SIFT的编码字典和特征编码
def extractECodebook(filenames,EcodeBookname,Ecodename,split=False,level=0,split_po=0):
    n = len(filenames)
    print(filenames[0])
    densift = dSIFT.DenseSIFT()
    E = None
    lenE=np.zeros(n,dtype=int)
    for i in range(n):
        print(i)
        img = cv2.imread(filenames[i])
        img=segmentation(img)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if split:
            if level==1:
                simgHSV=splitGEImage(imgGray)
                imgGray=simgHSV[split_po]
            elif level==2:
                simgHSV=splitGEImage(imgGray)
                chose_po=int(split_po/4)
                which_po=split_po%4
                imgGray=splitGEImage(simgHSV[chose_po])[which_po]
        imgEdge=cv2.Canny(imgGray,200,300)
        desE = densift.detectAndCompute(imgEdge[:, :])
        lenE[i]=len(desE)
        if E is None:
            E = desE
        else:
            E = np.concatenate((E, desE), axis=0)
    print('dSIFT complete')
    Ecodebook = getCodebook(E)
    saveCodebook(EcodeBookname,Ecodebook)

    sumE = 0
    ECode = np.zeros((n, 60))
    for i in range(n):
        print(i, "th LLC coding")
        if lenE[i]==0:
            ECode[i,:]=np.zeros(60)
        else:
            tempE = np.zeros((lenE[i], 60))
            for k in range(lenE[i]):
                tempE[k, :] = LLCoding(Ecodebook, E[sumE + k, :])
            sumE = sumE + lenE[i]
            ECode[i, :] = np.mean(tempE, axis=0)

    np.savetxt(Ecodename, ECode)
#提取灰度图的PHOW特征
def PHOW_G(img,Gcodebookname,split=False,split_po=0,level=0):
    Gcodebook=np.loadtxt(Gcodebookname)
    img = segmentation(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if split:
        if level == 1:
            simgHSV = splitGEImage(imgGray)
            imgGray = simgHSV[split_po]
        elif level == 2:
            simgHSV = splitGEImage(imgGray)
            chose_po = int(split_po / 4)
            which_po = split_po % 4
            imgGray = splitGEImage(simgHSV[chose_po])[which_po]

    #imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    densift=dSIFT.DenseSIFT()
    desG=densift.detectAndCompute(imgGray[:,:])
    GCode=np.zeros((len(desG),60))
    for i in range(len(desG)):
        #print(H[i,:].shape)
        GCode[i,:]=LLCoding(Gcodebook,desG[i,:])
    print('LLC coding complete')

    Gfinal=np.mean(GCode[:],axis=0).reshape(1,-1)

    return Gfinal
#提取edge-SIFT的PHOW特征
def PHOW_E(img,Ecodebookname,split=False,split_po=0,level=0):
    Ecodebook=np.loadtxt(Ecodebookname)
    img = segmentation(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if split:
        if level == 1:
            simgHSV = splitGEImage(imgGray)
            imgGray = simgHSV[split_po]
        elif level == 2:
            simgHSV = splitGEImage(imgGray)
            chose_po = int(split_po / 4)
            which_po = split_po % 4
            imgGray = splitGEImage(simgHSV[chose_po])[which_po]
    imgEdge = cv2.Canny(imgGray, 200, 300)
    #imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgEdge=cv2.Canny(imgGray,200,300)
    densift=dSIFT.DenseSIFT()
    desE=densift.detectAndCompute(imgEdge[:,:])
    ECode=np.zeros((len(desE),60))
    for i in range(len(desE)):
        #print(H[i,:].shape)
        ECode[i,:]=LLCoding(Ecodebook,desE[i,:])
    print('LLC coding complete')

    Efinal=np.mean(ECode[:],axis=0).reshape(1,-1)

    return Efinal
#保存灰度图的特征编码
def saveGcode(filenames,Gcodebookname,Gcodename,split=False,split_po=0,level=0):
    Filename=[]
    for filename in filenames:
        for f in filename:
            Filename.append(f)
    extractGCodebook(Filename,Gcodebookname,Gcodename,split,level,split_po)
#保存edge-SIFT的特征编码
def saveEcode(filenames,Ecodebookname,Ecodename,split=False,split_po=0,level=0):
    Filename=[]
    for filename in filenames:
        for f in filename:
            Filename.append(f)
    extractECodebook(Filename,Ecodebookname,Ecodename,split,level,split_po)