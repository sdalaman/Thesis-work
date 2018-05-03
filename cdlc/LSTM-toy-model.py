# toy model for LSTM classifier


import argparse
import numpy as np
from itertools import count
from collections import namedtuple
import os 

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
import torch.nn.init as init
import pickle
import glob
import time
import subprocess
from collections import namedtuple
import resource
import math

class LossValues(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.mean = []

class LSTMClassifier(nn.Module):
    def __init__(self,inputSize,lr_init,momentum):
        super(LSTMClassifier, self).__init__()
        self.inputSize = inputSize
        self.lr_init = lr_init
        self.momentum = momentum
        self.criterion = nn.BCELoss().cuda()
        self.lstm = nn.LSTMCell(self.inputSize, self.inputSize)
        self.fc = nn.Linear(inputSize,1)
        self.fc2 = nn.Linear(inputSize, 2)
        self.sgmd = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr_init, alpha=0.99, eps=1e-08, weight_decay=0, momentum=self.momentum,centered=False)

    def init_hidden(self,initrange):
        for ww in self.parameters():
            init.uniform(ww.data, -1 * initrange, initrange)

    def forwardOld(self, x, hidden):
        y = self.fc1(x)
        hx,cx = self.lstm(y,hidden)
        y = self.fc(hx)
        return y, hx,cx

    def forward(self,inp,hidden,step):
        t1 = torch.zeros(inp[0].size()[0],len(inp),inp[0].size()[1]).cuda()
        for i in range(len(inp)):
            for j in range(step):
                t1[j,i] = inp[i][j]
        for j in range(step):
            x = Variable(t1[j])
            hx,cx = self.lstm(x,hidden)
            hidden = (hx, cx)
        y1 = self.fc(hx)
        #yhat = y1.clone()
        yhat = y1
        return yhat

    def forward2(self,inp,hidden,step):
        t1 = torch.zeros(inp[0].size()[0],len(inp),inp[0].size()[1]).cuda()
        for i in range(len(inp)):
            for j in range(step):
                t1[j,i] = inp[i][j]
        for j in range(step):
            x = Variable(t1[j])
            hx,cx = self.lstm(x,hidden)
            hidden = (hx, cx)
        y1 = self.fc2(hx)
        #yhat = self.relu(hx.clone())
        yhat = F.log_softmax(y1)
        return yhat

def testModel(Tmodel,TestData,TestTarget,TinpSize,TSize):
    TallPositives=TallNegatives=0
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0

    for j in range(TSize):
        Tcx = Variable(torch.zeros(1, TinpSize).cuda())
        Thx = Variable(torch.zeros(1, TinpSize).cuda())
        Thidden = [Thx, Tcx]
        Tinp = [TestData[j].clone()]
        Tyhat = Tmodel.forward(Tinp, Thidden, Tinp[0].size()[0])
        pred = Tyhat.data.round()[0][0]
        #pred = Tyhat.data.max(1)[1]
        #pred = pred[0][0]
        if Tyhat.data.round()[0][0] > 0:
            pred = 1
        else:
            pred = 0

        if TestTarget[j] == 1:
            TallPositives += 1
            if pred == 1:
                TtruePos += 1
                TposPred += 1
            else:
                TfalseNeg += 1
                TnegPred += 1
        else:
            TallNegatives += 1
            if pred == 0:
                TtrueNeg += 1
                TnegPred += 1
            else:
                TfalsePos += 1
                TposPred += 1

        if pred == TestTarget[j]:
            Tcorrect += 1

        print("yhat:%f target:%f - correct:%d" % (pred, TestTarget[j], Tcorrect))

    return Tcorrect,TnegPred,TposPred,TtrueNeg,TtruePos,TfalseNeg,TfalsePos,TallPositives,TallNegatives

inputSize = 64
errors = []
lossVal = LossValues()
random.seed(9999)
trainSize = 50000  # 1000
testSize = 500    # 100
path = './'

trainData = []
testData = []
shuffle = np.random.permutation(trainSize)
tgTr = shuffle % 2
trainTarget = torch.Tensor(tgTr.data).cuda()
#trainTarget = torch.rand(trainSize).round().float().cuda()
shuffle = np.random.permutation(testSize)
tgTst = shuffle % 2
testTarget = torch.Tensor(tgTst.data).cuda()

step = 10
for cnt in range(trainSize):
    #step = random.randint(15, 100)
    tData = torch.rand(step, inputSize).float().cuda()
    for i in range(step):
        for j in range(10):
            tData[i,j] = random.uniform(0,0.5)
    trainData.append(tData)

for cnt in range(testSize):
    #step = random.randint(15, 100)
    tData = torch.rand(step, inputSize).float().cuda()
    for i in range(step):
        for j in range(10):
            tData[i,j] = random.uniform(0,0.5)
    testData.append(tData)

lrList = [1e-3,1e-4,1e-5]
mmList = [0.5,0.7]
#lrList = [1e-3]  # overfit 1e-3
#mmList = [0.7]   # overfit  0.7
maxEpoch = 10   # 100 10
batchSize = 50   # 1
results = []
init_weight = 0.0001  # 0.0001

docDiv = math.floor(trainSize / batchSize)
if trainSize % batchSize != 0:
    docDiv += 1

for lr in lrList:
    for mmt in mmList:
        model = LSTMClassifier(inputSize,lr,mmt).cuda()
        model.init_hidden(init_weight)
        lrStr  = ("%2.15f" % lr).rstrip('0')
        mmStr = ("%2.15f" % mmt).rstrip('0')
        lossVal = LossValues()
        errors = []
        allCnt = 0
        for epoch in range(maxEpoch):
            for j in range(docDiv):
                cx = Variable(torch.zeros(batchSize, inputSize).cuda())
                hx = Variable(torch.zeros(batchSize, inputSize).cuda())
                hidden = [hx, cx]
                inp = list(trainData[j*batchSize:(j+1)*batchSize])
                trgt = Variable(trainTarget[j * batchSize:(j + 1) * batchSize].cuda())
                allCnt += 1
                yhat = model.forward(inp,hidden,inp[0].size()[0])
                model.optimizer.zero_grad()
                error = ((yhat - trgt)*(yhat - trgt)).mean()
                #error = F.nll_loss(yhat, Variable(trgt.data.long().cuda()))
                print("epoch %d lr %s mmt %s train cnt %d error %f" % (epoch,lrStr,mmStr,j,error.data[0]))
                error.backward()
                model.optimizer.step()
                errors.append(error.data[0])
                lossVal.y.append(error.data[0])
                mean = torch.mean(torch.Tensor(errors).cuda())
                lossVal.mean.append(mean)

            if epoch % 50 == 0 and epoch != 0:
                correct, negPred, posPred, trueNeg, truePos, falseNeg, falsePos, allPositives, allNegatives = testModel(model, trainData, trainTarget, inputSize,trainSize)
                posRate = negRate = 0
                if allPositives != 0:
                    posRate = truePos / allPositives
                if allNegatives != 0:
                    negRate = trueNeg / allNegatives
                res = "train lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
                    lrStr, mmStr, maxEpoch, epoch+1, correct, trainSize, trueNeg, allNegatives, negRate, truePos,allPositives, posRate)
                results.append(res)

        correct,negPred,posPred,trueNeg,truePos,falseNeg,falsePos,allPositives,allNegatives = \
            testModel(model,trainData,trainTarget,inputSize,trainSize)
        posRate = negRate = 0
        if allPositives != 0:
            posRate = truePos / allPositives
        if allNegatives != 0:
            negRate = trueNeg / allNegatives
        res = "train lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
            lrStr, mmStr, maxEpoch, maxEpoch, correct, trainSize, trueNeg, allNegatives, negRate, truePos, allPositives,
            posRate)
        results.append(res)

        correct, negPred, posPred, trueNeg, truePos, falseNeg, falsePos, allPositives, allNegatives = \
            testModel(model, testData, testTarget, inputSize, testSize)
        posRate = negRate = 0
        if allPositives != 0:
            posRate = truePos / allPositives
        if allNegatives != 0:
            negRate = trueNeg / allNegatives
        res = "test lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
            lrStr, mmStr, maxEpoch,maxEpoch,correct,testSize,trueNeg,allNegatives,negRate,truePos,allPositives,posRate)
        results.append(res)

        lossVal.x = range(allCnt)
        fname = ("%sLSTM-toy-model.errors-%s-%s-%d-%d-%d.bin" % (path,lrStr,mmStr,maxEpoch,batchSize,step))
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()

for res in results:
    print(res)

print("end of program")
