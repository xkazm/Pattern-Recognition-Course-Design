import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from random import choice,shuffle
from numpy import array

def KMeansCluster(vectors,noofclusters):
    noofclusters=int(noofclusters)
    assert noofclusters<len(vectors)
    dim=len(vectors[0])
    print(dim)
    vector_indices=list(range(len(vectors)))
    shuffle(vector_indices)
    graph=tf.Graph()
    with graph.as_default():
        sess=tf.compat.v1.Session()
        centroids=[tf.Variable((vectors[vector_indices[i]])) for i in range(noofclusters)]
        centroid_value=tf.compat.v1.placeholder("float64",[dim])
        cent_assigns=[]
        for centroid in centroids:
            cent_assigns.append(tf.compat.v1.assign(centroid,centroid_value))
        assignments=[tf.Variable(0) for i in range(len(vectors))]
        #print(assignments)
        assignment_value=tf.compat.v1.placeholder("int32")
        cluster_assigns=[]
        for assignment in assignments:
            cluster_assigns.append(tf.compat.v1.assign(assignment,assignment_value))
        mean_input=tf.compat.v1.placeholder("float",[None,dim])
        mean_op=tf.reduce_mean(mean_input,0)
        v1=tf.placeholder("float",[dim])
        v2=tf.placeholder("float",[dim])
        euclid_dist=tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1,v2),2)))
        centroid_distances=tf.placeholder("float",[noofclusters])
        cluster_assignment=tf.argmin(centroid_distances,0)
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        noofiterations=20
        for iteration_n in range(noofiterations):
            for vector_n in range(len(vectors)):
                vect=vectors[vector_n]
                distances=[sess.run(euclid_dist,feed_dict={v1:vect,v2:sess.run(centroid)}) for centroid in centroids]
                assignment=sess.run(cluster_assignment,feed_dict={centroid_distances:distances})
                sess.run(cluster_assigns[vector_n],feed_dict={assignment_value:assignment})
            for cluster_n in range(noofclusters):
                assigned_vects=[vectors[i] for i in range(len(vectors)) if sess.run(assignments[i])==cluster_n]
                new_location=sess.run(mean_op,feed_dict={mean_input:array(assigned_vects)})
                sess.run(cent_assigns[cluster_n],feed_dict={centroid_value:new_location})
        centroids=sess.run(centroids)
        assignments=sess.run(assignments)
        return centroids,assignments
"""
sampleNo=10
#mu=3
mu=np.array([[1,5]])
Sigma=np.array([[1,0.5],[1.5,3]])
R=cholesky(Sigma)
srcdata=np.dot(np.random.randn(sampleNo,2),R)+mu
print(srcdata)
plt.plot(srcdata[:,0],srcdata[:,1],'bo')

k=4
center,result=KMeansCluster(srcdata,k)
print(center)

res={"x":[],"y":[],"kmeans_res":[]}
for i in range(len(result)):
    res["x"].append(srcdata[i][0])
    res["y"].append(srcdata[i][1])
    res["kmeans_res"].append(result[i])
pd_res=pd.DataFrame(res)
sns.lmplot("x","y",data=pd_res,fit_reg=False,size=5,hue="kmeans_res")
plt.show()
"""