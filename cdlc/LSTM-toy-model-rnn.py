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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self,initrange):
        for ww in self.parameters():
            init.uniform(ww.data, -1 * initrange, initrange)

class modelPrm(object):
    def __init__(self):
        self.inputSize = 0
        self.hiddenSize = 0
        self.numLayers = 0
        self.numClasses = 0
        self.lr = 0
        self.mmt = 0
        self.maxEpoch = 0
        self.trainSize = 0
        self.testSize = 0
        self.sequenceLength = 0
        self.batchSize = 0
        self.init_weight = 0


def trainModel(trainData,trainTarget,testData,testTarget,prms,results):
    model = RNN(prms.inputSize, prms.hiddenSize, prms.numLayers, prms.numClasses).cuda()
    model.init_hidden(prms.init_weight)
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=mmt,centered=True)
    optimizer = optim.SGD(model.parameters(), lr=prms.lr, momentum=prms.mmt)
    lrStr = ("%2.15f" % prms.lr).rstrip('0')
    mmStr = ("%2.15f" % prms.mmt).rstrip('0')
    lossVal = LossValues()
    errors = []
    allCnt = 0
    res = "lr %s mmt %s maxEpoch %d sequence length %d hidden size %d num of layers %d init weight %f batch size %d" % (
        lrStr, mmStr, prms.maxEpoch,prms.sequenceLength ,prms.hiddenSize,prms.numLayers,prms.init_weight,prms.batchSize)
    results.append(res)

    for epoch in range(prms.maxEpoch):
        for batchNum in range(prms.trainSize):
            allCnt += 1
            inp = trainData[batchNum].clone()
            trgt = trainTarget[i].clone()
            inp = Variable(inp.view(-1, prms.sequenceLength, prms.inputSize).cuda())
            labels = Variable(trgt.cuda())
            optimizer.zero_grad()
            outputs = model(inp)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print("epoch %d lr %s mmt %s train batch %d error %f" % (epoch, lrStr, mmStr, batchNum, loss.data[0]))
            errors.append(loss.data[0])
            lossVal.y.append(loss.data[0])
            mean = torch.mean(torch.Tensor(errors).cuda())
            lossVal.mean.append(mean)

        if epoch % 50000 == 0 and epoch != 0:
            correct, negPred, posPred, trueNeg, truePos, falseNeg, falsePos, allPositives, allNegatives = testModel(
                model, trainData, trainTarget,prms)
            posRate = negRate = 0
            if allPositives != 0:
                posRate = truePos / allPositives
            if allNegatives != 0:
                negRate = trueNeg / allNegatives
            res = "train lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
                lrStr, mmStr, prms.maxEpoch, epoch + 1, correct, prms.trainSize, trueNeg, allNegatives, negRate, truePos,
                allPositives, posRate)
            results.append(res)

    correct, negPred, posPred, trueNeg, truePos, falseNeg, falsePos, allPositives, allNegatives = \
        testModel(model, trainData, trainTarget,prms)
    posRate = negRate = 0
    if allPositives != 0:
        posRate = truePos / allPositives
    if allNegatives != 0:
        negRate = trueNeg / allNegatives
    res = "train lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
        lrStr, mmStr, prms.maxEpoch, prms.maxEpoch, correct, prms.trainSize, trueNeg, allNegatives, negRate, truePos, allPositives,
        posRate)
    results.append(res)

    correct, negPred, posPred, trueNeg, truePos, falseNeg, falsePos, allPositives, allNegatives = \
        testModel(model, testData, testTarget,prms)
    posRate = negRate = 0
    if allPositives != 0:
        posRate = truePos / allPositives
    if allNegatives != 0:
        negRate = trueNeg / allNegatives
    res = "test lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNeg/allNegatives:%d/%d=%f  truePos/allPositives:%d/%d=%f" % (
        lrStr, mmStr, prms.maxEpoch, prms.maxEpoch, correct, prms.testSize, trueNeg, allNegatives, negRate, truePos, allPositives,
        posRate)
    results.append(res)

    lossVal.x = range(allCnt)
    fname = ("%sLSTM-toy-model-rnn-errors-%s-%s-%d-%d-%d-%d-%d.bin" % (
    path, lrStr, mmStr, prms.maxEpoch, prms.batchSize, prms.sequenceLength, prms.hiddenSize, prms.numLayers))
    fh = open(fname, 'wb')  # Save errors
    pickle.dump(lossVal, fh)
    fh.close()
    return model,results


def testModel(Tmodel,TestData,TestTarget,prms):
    TallPositives=TallNegatives=0
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0

    TestData = TestData.view(-1, prms.sequenceLength, prms.inputSize).cuda()
    TestTarget = TestTarget.view(-1).cuda()

    for j in range(TestData.size()[0]):
        Tinp = TestData[j].clone()
        Tinp = Variable(Tinp.view(-1, prms.sequenceLength, prms.inputSize).cuda())
        Tyhat = Tmodel(Tinp)
        pred = torch.max(Tyhat.data, 1)
        pred = pred[1][0][0]

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

path = './'
results = []
errors = []
lossVal = LossValues()
mdlPrms = modelPrm()
random.seed(9999)

mdlPrms.trainSize = 1000  # 1000
mdlPrms.testSize = 100    # 100
# Hyper Parameters
mdlPrms.sequenceLength = 10 # 20
mdlPrms.inputSize = 64 # 64
mdlPrms.numClasses = 2 # 2
mdlPrms.batchSize = 20 # 20
mdlPrms.maxEpoch = 10 # 500
mdlPrms.init_weight = 1  # 1

hiddenSizeList = [1024] # 128
numLayersList = [3] # 2
lrList = [1e-3] # 1e-3
mmList = [0.7] # 0.7

trainTarget = torch.Tensor(mdlPrms.trainSize,mdlPrms.batchSize).long().cuda()
for i in range(mdlPrms.trainSize):
    shuffle = np.random.permutation(mdlPrms.batchSize)
    tgTr = shuffle % mdlPrms.numClasses
    trainTarget[i] = torch.Tensor(tgTr.data).long().cuda()

shuffle = np.random.permutation(mdlPrms.testSize)
tgTst = shuffle % mdlPrms.numClasses
testTarget = torch.Tensor(tgTst.data).long().cuda()

trainData = torch.rand(mdlPrms.trainSize,mdlPrms.batchSize,mdlPrms.sequenceLength,mdlPrms.inputSize).float().cuda()
for cnt in range(mdlPrms.trainSize):
    for b in range(mdlPrms.batchSize):
        for i in range(mdlPrms.sequenceLength):
            for j in range(10):
                trainData[cnt,b,i,j] = random.uniform(0,0.5)

testData = torch.rand(mdlPrms.testSize,mdlPrms.sequenceLength,mdlPrms.inputSize).float().cuda()
for cnt in range(mdlPrms.testSize):
    tData = torch.rand(mdlPrms.sequenceLength,mdlPrms.inputSize).float().cuda()
    for i in range(mdlPrms.sequenceLength):
        for j in range(10):
            testData[cnt,i,j] = random.uniform(0,0.5)

for lr in lrList:
    for mmt in mmList:
        for hiddenSize in hiddenSizeList:
            for numLayers in numLayersList:
                mdlPrms.lr = lr
                mdlPrms.mmt = mmt
                mdlPrms.hiddenSize = hiddenSize
                mdlPrms.numLayers = numLayers
                model,results = trainModel(trainData,trainTarget,testData,testTarget,mdlPrms,results)

fname = ("LSTM-toy-model-rnn-errors-out.txt")
fh = open(fname, 'w')  # Save errors
for res in results:
    print(res)
    fh.write(res)
    fh.write("\n")
fh.close()

print("end of program")
