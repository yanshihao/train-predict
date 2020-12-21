#!/usr/bin/env python
# coding: utf-8

from utils import readCsvFile, testPaths, dataIter, DataIter, srcIndex,trgIndex,mu,sig,homeDirectory
import torch
from model import model,device
import matplotlib.pyplot as plt
import numpy as np

model.load_state_dict(torch.load(homeDirectory+'my-model.pt'))

keyPointsLocationLists= []
keyPointsLocations = readCsvFile(homeDirectory+'dataset/'+ testPaths[30])
# 读取一条文件并画图显示轨迹
nums = len(keyPointsLocations)//12
numIndex = 12*np.array([x for x in range(nums)])
keyPointsLocations = keyPointsLocations[numIndex,:]

fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111)
ax.plot(keyPointsLocations[:,0],keyPointsLocations[:,2],'ro',label = 'trunk_src')            
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ran = [xlim[1]-xlim[0],ylim[1]-ylim[0]]
loc = [(xlim[1]+xlim[0])/2,(ylim[1]+ylim[0])/2]
maxRan = 3600
ran = [loc[0]-maxRan/2, loc[0]+maxRan/2,loc[1]-maxRan/2, loc[1]+maxRan/2]
ax.set_xlim(ran[0:2])
ax.set_ylim(ran[2:4])
ax.legend()
fig.show()

# 获取中心点，并将数据进行归一化
centerLocs = []

for i in range(nums-20):
    keyPointsLocation = keyPointsLocations[i:i+20,:].transpose()
    centerLoc = keyPointsLocation[:,9]
    centerLocs.append(centerLoc)
    keyPointsLocation = (keyPointsLocation- centerLoc[:,np.newaxis]-mu[:,np.newaxis])/sig[:,np.newaxis]
    keyPointsLocationLists.append(keyPointsLocation)


# 送入网络测试，并绘图
t0 = np.ones([18,1],dtype = np.float32)
t1 = np.linspace(0,1.7,18).astype(np.float32).reshape((-1,1))
t2= t1*t1
t3= t2*t1
T = np.concatenate([t0,t1,t2,t3],axis = 1)
T_inv = np.linalg.pinv(T)
valid_iterator = DataIter(keyPointsLocationLists[30:],device,1, centerLocs)
for i, batch in enumerate(valid_iterator):
    src = batch.src
    trg = batch.trg
    model.eval()
    output = model(src, trg, 0)
    def real_trajectory_plot(src, trg, cents):
        src_reals = src.cpu().numpy()
        trg_reals = trg.cpu().numpy()
        out_reals = output.cpu().detach().numpy()
        cents = cents.cpu().numpy()
        _,batch_size, _ = src_reals.shape
        
        for i in range(batch_size):
            src_real = src_reals[:,i,:]*sig[srcIndex]+ mu[srcIndex] + cents[i,srcIndex]
            trg_real = trg_reals[:,i,:]*sig[trgIndex] + mu[trgIndex] + cents[i,trgIndex]
            out_real = out_reals[:,i,:]*sig[trgIndex] + mu[trgIndex] + cents[i,trgIndex]
            
            plot_train = np.concatenate([src_real[:,[0,1]],out_real],axis= 0)
            Weights = np.matmul(T_inv,plot_train)
            plot_real = np.matmul(T,Weights)
            fig = plt.figure(figsize=[8,6])
            ax = fig.add_subplot(111)
            ax.plot(src_real[:,0],src_real[:,1],'ro',label = 'trunk_src')
            ax.plot(trg_real[:,0],trg_real[:,1],'go',label = 'trunk_trg')
            ax.plot(out_real[:,0],out_real[:,1],'bo',label = 'trunk_out')
            
            ax.plot(plot_real[:,0],plot_real[:,1],linestyle = '--',color = 'brown',label = 'plot_real')
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ran = [xlim[1]-xlim[0],ylim[1]-ylim[0]]
            loc = [(xlim[1]+xlim[0])/2,(ylim[1]+ylim[0])/2]
            maxRan = 2000
            ran = [loc[0]-maxRan/2, loc[0]+maxRan/2,loc[1]-maxRan/2, loc[1]+maxRan/2]
            ax.set_xlim(ran[0:2])
            ax.set_ylim(ran[2:4])
            ax.legend()
            fig.show()
    if  i >10 and i < 30 : 
        real_trajectory_plot(src, trg, batch.cent)
#为了防止程序直接退出，接受一个字符后才结束
raw_input()

