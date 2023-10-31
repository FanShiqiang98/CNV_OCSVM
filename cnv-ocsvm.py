import os.path

import numpy as np
import pandas as pd

#xmwans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


from sklearn.svm import OneClassSVM
from preprocess import preProcessData


def xMeans(scores):
    amount_initial_centers = 2
    sample=[]
    for i in range(len(scores)):
        sample.append([i,scores[i]])
    initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
    xmeans_instance = xmeans(sample, initial_centers, 2)
    xmeans_instance.process()
    a=xmeans_instance.get_clusters()
    centers=xmeans_instance.get_centers()
    tag=0
    if centers[0][1]>centers[1][1]:
        tag=1
    res=[]
    for i in range(len(a[tag])):
        res.append(scores[a[tag][i]])
    return centers,a[tag]


def run(bamFilePath,baseSavePath,refPath,nu,gamma,gtPath=None):
    if os.path.exists(baseSavePath)==False:
        os.mkdir(baseSavePath)
    all_chr,all_start,all_end,all_RD,mode= preProcessData(refPath, bamFilePath,1)


    trainData = all_RD.reshape((-1, 1))

    # 使用sklearn
    model = OneClassSVM(nu=nu, gamma=gamma,kernel='rbf')
    model.fit(trainData)
    res = model.decision_function(trainData)

    #阈值确定

    center,cluster=xMeans(res)
    if center[0][1]<center[1][1]:
        lower=center[0][1]
    else:
        lower=center[1][1]

    cnvRegion = []
    result_start=[]
    result_end=[]
    result_type=[]
    for i in range(len(res)):
        if res[i]<lower:
            if all_RD[i]>mode:
                state='duplication'
            else:
                state='deletion'
            result_start.append(all_start[i])
            result_end.append(all_end[i])
            result_type.append(state)
            cnvRegion.append([all_start[i],all_end[i],state])

    df=pd.DataFrame(cnvRegion,columns=['binStart','binEnd','state'])
    fileName=bamFilePath.split('/')[-1]
    df.to_csv(f"{baseSavePath}/{fileName}.csv")
    return df



if __name__=='__main__':
    gamma=0.00001
    con=0.1
    nu=0.99
    finalResult=[]

    bamFilePath = F'./realData/NA12878.chrom21.SLX.maq.SRP000032.2009_07.bam'
    baseSavePath = './realData/'
    refPath = "./"
    df=run(bamFilePath,'./realData/', refPath, nu, gamma)
    df.to_csv(f'./realData/OCSVM_NA12878.csv')

