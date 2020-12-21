import random
import os
import math
import torch
homeDirectory = '/home/nvidia/test_dir/train-predict/'
paths = ['Turn left 90 HJ 2019-04-12 04.00.40 PM_038.csv',
'Turn left 90 HJ 2019-04-12 04.00.40 PM_039.csv',
'Turn left 90  LJL 2019-04-16 07.59.36 PM.csv',
'Turn left 90  LJL 2019-04-16 07.59.36 PM_003.csv',
'Turn left 90  LJL 2019-04-16 07.59.36 PM_004.csv',
'Turn left 90 RHG 2019-04-15 07.37.09 PM.csv',
'Turn left 90 RHG 2019-04-15 07.37.09 PM_001.csv',
'Turn left 90 RHG 2019-04-15 07.37.09 PM_002.csv',
'Turn left 90 RHG 2019-04-15 07.37.09 PM_003.csv',
'Turn left 90 RHG 2019-04-15 07.37.09 PM_005.csv',
'Turn left 90 RHG  2019-04-15 07.37.09 PM_004.csv',
'Turn left 90 WS 2019-04-15 08.05.32 PM.csv',
'Turn left 90 WS 2019-04-15 08.05.32 PM_001.csv',
'Turn left 90 WS 2019-04-15 08.05.32 PM_002.csv',
'Turn left 90 WS 2019-04-15 08.05.32 PM_003.csv',
'Turn  left  90 CMT 2019-04-17 09.31.11 AM_016.csv',
'Turn  left  90 CMT 2019-04-17 09.31.11 AM_017.csv',
'Turn  left  90 CMT 2019-04-17 09.31.11 AM_018.csv',
'Turn  left  90 CMT 2019-04-17 09.31.11 AM_019.csv',
'Turn  left  90 CMT 2019-04-17 09.31.11 AM_020.csv',
'Turn right 90 HJ 2019-04-12 04.00.40 PM_040.csv',
'Turn right 90 HJ 2019-04-12 04.00.40 PM_041.csv',
'Turn right 90  LJL  2019-04-16 08.04.11 PM.csv',
'Turn right 90  LJL  2019-04-16 08.04.11 PM_001.csv',
'Turn right 90  LJL  2019-04-16 08.04.11 PM_002.csv',
'Turn right 90  LJL  2019-04-16 08.04.11 PM_003.csv',
'Turn right 90  LJL  2019-04-16 08.04.11 PM_004.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM_001.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM_002.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM_003.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM_004.csv',
'Turn right 90 RHG 2019-04-15 07.41.59 PM_005.csv',
'Turn right 90 WS 2019-04-15 08.05.32 PM_016.csv',
'Turn right 90 WS 2019-04-15 08.05.32 PM_017.csv',
'Turn right 90 WS 2019-04-15 08.05.32 PM_018.csv',
'Turn right 90 WS 2019-04-15 08.05.32 PM_019.csv',
'Turn right 90 WS 2019-04-15 08.05.32 PM_020.csv',
'Turn right 90 ZY 2019-04-17 10.19.24 AM_005.csv',
'Turn right 90 ZY 2019-04-17 10.19.24 AM_006.csv',
'walk straight forward CMT 2019-04-17 09.31.11 AM.csv',
'walk straight forward CMT 2019-04-17 09.31.11 AM_001.csv',
'walk straight forward CMT 2019-04-17 09.31.11 AM_002.csv',
'walk straight forward CMT 2019-04-17 09.31.11 AM_003.csv',
'walk straight forward CMT 2019-04-17 09.31.11 AM_004.csv',
'Walk straight forward  HJ 2019-04-12 04.00.40 PM_019.csv',
'Walk straight forward  HJ 2019-04-12 04.00.40 PM_020.csv',
'Walk straight forward LJL 2019-04-16 07.49.43 PM.csv',
'Walk straight forward LJL 2019-04-16 07.49.43 PM_001.csv',
'Walk straight forward RHG 2019-04-15 07.33.32 PM.csv',
'Walk straight forward RHG 2019-04-15 07.33.32 PM_001.csv',
'Walk straight forward RHG 2019-04-15 07.33.32 PM_002.csv',
'Walk straight forward RHG 2019-04-15 07.33.32 PM_003.csv',
'Walk straight forward RHG 2019-04-15 07.33.32 PM_004.csv',
'Walk straight forward WS 2019-04-15 07.56.11 PM.csv',
'Walk straight forward WS 2019-04-15 07.56.11 PM_001.csv',
'Turn left 90 WS 2019-04-15 08.05.32 PM_004.csv',
'Turn left 90 ZY 2019-04-17 10.07.12 AM.csv',
'Turn left 90 ZY 2019-04-17 10.07.12 AM_001.csv',
'Turn left 90 ZY 2019-04-17 10.07.12 AM_002.csv',
'Turn left 90 ZY 2019-04-17 10.07.12 AM_003.csv',
'Turn left 90 ZY 2019-04-17 10.07.12 AM_004.csv',
'Turn right 90 ZY 2019-04-17 10.19.24 AM_007.csv',
'Turn right 90 ZY 2019-04-17 10.19.24 AM_008.csv',
'Turn right 90 ZY 2019-04-17 10.19.24 AM_009.csv',
'Turn  right  90 CMT 2019-04-17 09.31.11 AM_026.csv',
'Turn  right  90 CMT 2019-04-17 09.31.11 AM_027.csv',
'Turn  right  90 CMT 2019-04-17 09.31.11 AM_028.csv',
'Turn  right  90 CMT 2019-04-17 09.31.11 AM_029.csv',
'Turn  right  90 CMT 2019-04-17 09.31.11 AM_030.csv',
'Walk straight forward WS 2019-04-15 07.56.11 PM_002.csv',
'Walk straight forward WS 2019-04-15 07.56.11 PM_003.csv',
'Walk straight forward WS 2019-04-15 07.56.11 PM_004.csv',
'walk straight forward ZY 2019-04-17 09.31.11 AM_005.csv',
'walk straight forward ZY 2019-04-17 09.31.11 AM_006.csv',
'walk straight forward ZY 2019-04-17 09.31.11 AM_007.csv',
'walk straight forward ZY 2019-04-17 09.31.11 AM_009.csv',
'Walk straight forward ZY 2019-04-17 09.31.11 AM_008.csv',
         
'Turn  left  180 CMT 2019-04-17 09.31.11 AM_021.csv',
"Turn  left  180 CMT 2019-04-17 09.31.11 AM_022.csv",
"Turn  left  180 CMT 2019-04-17 09.31.11 AM_023.csv",
"Turn  left  180 CMT 2019-04-17 09.31.11 AM_025.csv",
"Turn left 180  LJL 2019-04-16 08.02.25 PM_004.csv",
"Turn left 180 ZY 2019-04-17 10.19.24 AM.csv",
"Turn left 180 ZY 2019-04-17 10.19.24 AM_001.csv",
"Turn left 180 ZY 2019-04-17 10.19.24 AM_002.csv",
"Turn left 180 ZY 2019-04-17 10.19.24 AM_003.csv",
"Turn left 180 ZY 2019-04-17 10.19.24 AM_004.csv",
"Turn right 180 CMT 2019-04-17 09.31.11 AM_031.csv",
"Turn right 180 CMT 2019-04-17 09.31.11 AM_032.csv",
"Turn right 180 CMT 2019-04-17 09.31.11 AM_033.csv",
"Turn right 180 CMT 2019-04-17 09.31.11 AM_034.csv",
"Turn right 180 CMT 2019-04-17 09.31.11 AM_035.csv",
"Turn right 180 WS 2019-04-15 08.05.32 PM_021.csv",
"Turn right 180 WS 2019-04-15 08.05.32 PM_022.csv",
"Turn right 180 WS 2019-04-15 08.05.32 PM_023.csv",
"Turn right 180 WS 2019-04-15 08.05.32 PM_024.csv",
"Turn right 180 ZY 2019-04-17 10.19.24 AM_011.csv",
"Walk straight lateral  left ZY  2019-04-17 10.03.35 AM_005.csv",
"Walk straight lateral  left ZY  2019-04-17 10.03.35 AM_006.csv",
"Walk straight lateral  left ZY  2019-04-17 10.03.35 AM_007.csv",
"Walk straight lateral  left ZY  2019-04-17 10.03.35 AM_008.csv",
"Walk straight lateral  left ZY  2019-04-17 10.03.35 AM_009.csv",
"Walk straight lateral  right ZY  2019-04-17 10.03.35 AM_001.csv",
"Walk straight lateral  right ZY  2019-04-17 10.03.35 AM_002.csv",
"Walk straight lateral  right ZY  2019-04-17 10.03.35 AM_004.csv",
"Walk straight lateral left HJ 2019-04-12 04.00.40 PM_027.csv",
"Walk straight lateral left HJ 2019-04-12 04.00.40 PM_028.csv",
"Walk straight lateral left HJ 2019-04-12 04.00.40 PM_029.csv",
"Walk straight lateral left LJL 2019-04-16 07.55.40 PM_002.csv",
"Walk straight lateral right LJL 2019-04-16 07.49.43 PM_005.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_005.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_006.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_007.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_008.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_009.csv",
"walk straight lateral right CMT 2019-04-17 09.31.11 AM_010.csv"]

# shuffle the paths

testPaths  = []
trainPaths = []
if os.path.exists(homeDirectory+"testPath.txt") and os.path.exists(homeDirectory+"trainPath.txt"):
    fileTest = open(homeDirectory+'testPath.txt',mode = 'r')
    testPathsLines = fileTest.readlines()
    for testPathsLine in testPathsLines:
        testPath = testPathsLine.strip('\n')
        testPaths.append(testPath)
    
    fileTrain = open(homeDirectory+'trainPath.txt',mode = 'r')
    trainPathsLines = fileTrain.readlines()
    for trainPathsLine in trainPathsLines:
        trainPath = trainPathsLine.strip('\n')
        trainPaths.append(trainPath)
else:
    TRAIN_SET_RAIDO = 0.7
    numPaths = len(paths)
    numTrainPaths = int(math.floor(numPaths*TRAIN_SET_RAIDO))
    testPaths = paths[numTrainPaths:]
    trainPaths = paths[:numTrainPaths]
    # save 
    fileTest = open(homeDirectory+'testPath.txt', mode = 'w')
    fileTrain = open(homeDirectory+'trainPath.txt', mode = 'w')
    for testPath in paths[numTrainPaths:]:
        fileTest.write(testPath)
        fileTest.write('\n')
    fileTest.close()

    for trainPath in paths[:numTrainPaths]:
        fileTrain.write(trainPath)
        fileTrain.write('\n')
    fileTrain.close()

import csv
import numpy as np

# readCsvFile
# in:  csvFile
# out: keyPointsLocation, the shape is [times, 9], 9 is for Trunk XYZ,RTip XYZ, LTip XYZ
def readCsvFile(csvFile):
    with open(csvFile, "r") as csvfile:
        reader = csv.reader(csvfile)
        transforms = []
        keyPointsLocation =[]
        for t, line in enumerate(reader):
            if t >= 5:
                keyPointsLocation.append([float(number) for number in line[2:14]])
            elif t == 1:
                keyPoints = line[2:14]
            elif t == 4:
                axis = line[2:14]
                for n, keyPoint in enumerate(keyPoints):
                    transform = 0
                    if keyPoint == 'Trunk Front':
                        transform = 0
                    if keyPoint == 'RTip':
                        transform = 3
                    if keyPoint == 'LTip':
                        transform = 6
                    if keyPoint == 'Trunk Back':
                        transform = 9
                    if axis[n] == 'Y':
                        transform += 1
                    if axis[n] == 'Z':
                        transform += 2
                    transforms.append(transform)
        keyPointsLocation = np.array(keyPointsLocation)
        transforms = [transforms.index(i) for i in range(12)]
        keyPointsLocation =  keyPointsLocation[:,transforms]
        keyPointsLocation[:,0:3] = (keyPointsLocation[:,0:3]+ keyPointsLocation[:,9:12])/2
        return keyPointsLocation[:,0:9]

class dataIter():
    def __init__(self,src, trg, cent):
        self.src = src
        self.trg = trg
        self.cent = cent

srcIndex = [0,2,3,5,6,8]
trgIndex = [0,2]
batchSize = 2

mu = np.load(homeDirectory+'mu.npy')
sig = np.load(homeDirectory+'sig.npy')

def DataIter(trainLocationDatas,device,batch, centerLocs):
    numTrainData = len(trainLocationDatas)
    times = numTrainData//batch
    f = trainLocationDatas[0].shape[0]
    data = []
    for time in range(times):
        src = np.empty([10,batch,len(srcIndex)],dtype = float)
        for i in range(batch): src[:,i,:] = trainLocationDatas[batch*time + i][srcIndex,0:10].T
        trg = np.empty([8,batch,len(trgIndex)],dtype = float)
        for i in range(batch): trg[:,i,:] = trainLocationDatas[batch*time + i][trgIndex,10:18].T
        cent = np.empty([batch,f],dtype = float)
        for i in range(batch): cent[i,:] = centerLocs[batch*time + i]
        yield dataIter(torch.tensor(src,dtype=torch.float32).to(device),
                           torch.tensor(trg,dtype=torch.float32).to(device),
                           torch.tensor(cent,dtype=torch.float32).to(device))
