#!/usr/bin/env python
# coding: utf-8

# # 时序序列预测
# 本文要解决的问题主要是根据一段时间内的轨迹来预测后一段时间内的轨迹。此问题与机器翻译问题类似，给出两个对应的时间序列来生成。
# In[1]:


import torch
import numpy as np
import time
print(torch.__version__)


# In[2]:


import random
from utils import paths, readCsvFile, homeDirectory

import math

# 打乱顺序
random.shuffle(paths)
locationDatas = []
TRAIN_SET_RAIDO = 0.7
numPaths = len(paths)
numTrainPaths = math.floor(numPaths*TRAIN_SET_RAIDO)
numTestPaths = numPaths - numTrainPaths
numTrainData = 0
numTestData  = 0

for num, path in enumerate(paths):
    locationData = readCsvFile(homeDirectory+'dataset/'+ path)
    nums = len(locationData)//12
    numIndex = 12*np.array([x for x in range(nums)])
    locationData = locationData[numIndex,:]
    for i in range(nums//10-1):
        locationDatas.append(locationData[i*10:i*10+20,:].transpose())
    if num < numTrainPaths:
        numTrainData += nums//10
    else:
        numTestData  += nums//10

len(locationDatas)
print("the numbers of Paths is : %d"%(numPaths))
print("the numbers of TrainPaths is : %d"%(numTrainPaths))
print("the numbers of TestPaths is : %d"%(numTestPaths))
print("the numbers of TrainData is : %d"%(numTrainData))
print("the numbers of TestData is : %d"%(numTestData))


# In[5]:

# 记录中心位置
centerLocs = []
import math
random.shuffle(locationDatas[0:numTrainData])
random.shuffle(locationDatas[numTrainData:])
for i, locationData in enumerate(locationDatas):
    centerLoc = locationData[:,9]
    locationDatas[i] = locationData - locationData[:,9][:,np.newaxis]
    centerLocs.append(centerLoc)

combineData = locationDatas[0]


# 计算方差和均值
for i in range(numTrainData-1):
    combineData = np.concatenate((combineData,locationDatas[i+1]),axis = 1)
    
mu = np.mean(combineData,1)
sig = np.std(combineData,1)

for i, locationData in enumerate(locationDatas):
    locationDatas[i] = (locationDatas[i]-mu[:,np.newaxis])/sig[:,np.newaxis]

print("the mean of the conbineData is:")
print(mu)
print("the std of the conbineData is:")
print(sig)
print("so the final shape of the conbineData is:")
print(combineData.shape)

# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim
from model import model, device
from utils import dataIter, DataIter, srcIndex,trgIndex

batchSize = 2
# In[15]:

# 初始化权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)


# In[16]:
optimizer = optim.Adam(model.parameters())


# In[17]:
criterion = nn.MSELoss()


# In[18]:
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    len_iterator = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        len_iterator += 1
        
    return epoch_loss / len_iterator


# In[19]:


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    len_iterator = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
            len_iterator += 1
            
    return epoch_loss / len_iterator


# In[20]:


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[21]:
N_EPOCHS = 40
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_iterator = DataIter(locationDatas[0:numTrainData],device,2, centerLocs)
    valid_iterator = DataIter(locationDatas[numTrainData:],device,2, centerLocs)
    # 训练
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # 验证
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'my-model-test.pt')
    
    print("Epoch:", epoch+1, "| Time:", epoch_mins, "m", epoch_secs,"s")
    print("\tTrain Loss:", train_loss )
    print("\tVal Loss:", valid_loss )
# 还差一个测试

# In[22]:


print('best_valid_loss is ')
print(best_valid_loss)

# 将训练数据和测试数据存储在txt文件中，将mu和sig存在.npy文件中
fileTest = open(homeDirectory+'testPath.txt', mode = 'w')
fileTrain = open(homeDirectory+'trainPath.txt', mode = 'w')
for testPath in paths[int(numTrainPaths):]:
    fileTest.write(testPath)
    fileTest.write('\n')
fileTest.close()

for trainPath in paths[:int(numTrainPaths)]:
    fileTrain.write(trainPath)
    fileTrain.write('\n')
fileTrain.close()

np.save(homeDirectory+'mu.npy', mu)
np.save(homeDirectory+'sig.npy', sig)
