import numpy as np
from scipy import stats

Ea=np.array([77.5,77.5,75.5,75.0,92.5,75.5,85.0,87.5,56.25,98.61,73.61,50.0,64.58,68.75,89.58,72.92])
#Eb=np.array([75.0,80.5,74.0,83.0,93.5,77.5,72.0,92.36,62.5,97.22,74.31,52.08,70.83,66.67,90.97,80.56])
Eb=np.array([76.0,79.5,75.0,83.0,91.5,74.5,85.5,85.42,64.58,99.31,65.72,45.14,67.36,69.44,91.67,83.33])

meana=np.mean(Ea)
meanb=np.mean(Eb)

stda=np.std(Ea)
stdb=np.std(Eb)

nobs1=len(Ea)
nobs2=len(Eb)

modified_stda=np.sqrt(np.float32(nobs1)/np.float32(nobs1)-1)*stda
modified_stdb=np.sqrt(np.float32(nobs2)/np.float32(nobs2-1))*stdb

(statistic,pvalue)=stats.ttest_ind_from_stats(meana,modified_stda,nobs1,meanb,modified_stdb,nobs2)
print(pvalue)