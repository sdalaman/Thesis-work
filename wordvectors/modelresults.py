
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import pickle
import datetime
import numpy as np

def floatToStr(frm,prm):
    return (frm % prm).rstrip('0')

class LossValues(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.mean = []

class ModelStats(object):
    def __init__(self):
        self.correct = []
        self.fold = []
        self.epoch = []
        self.predPositives = []
        self.predNegatives = []
        self.trueNegatives = []
        self.truePositives = []
        self.falseNegatives = []
        self.falsePositive = []
        self.allPositives = []
        self.allNegatives = []
        self.all = []
        self.posRate = []
        self.negRate = []
        self.precision = []
        self.recall = []
        self.f1Score = []
        self.score = []
        self.accuracy = []
        self.errorRate = []

    def addStat(self,fold,epoch,mdlRes):
        self.correct.append(mdlRes.correct)
        self.fold.append(fold)
        self.epoch.append(epoch)
        self.predPositives.append(mdlRes.predPositives)
        self.predNegatives.append(mdlRes.predNegatives)
        self.trueNegatives.append(mdlRes.trueNegatives)
        self.truePositives.append(mdlRes.truePositives)
        self.falseNegatives.append(mdlRes.falseNegatives)
        self.falsePositive.append(mdlRes.falsePositive)
        self.allPositives.append(mdlRes.allPositives)
        self.allNegatives.append(mdlRes.allNegatives)
        self.all.append(mdlRes.all)
        self.posRate.append(mdlRes.posRate)
        self.negRate.append(mdlRes.negRate)
        self.precision.append(mdlRes.precision)
        self.recall.append(mdlRes.recall)
        self.f1Score.append(mdlRes.f1Score)
        self.score.append(mdlRes.score)
        self.accuracy.append(mdlRes.accuracy)
        self.errorRate.append(mdlRes.errorRate)


class ModelPrm(object):
    def __init__(self):
        self.inputSize = 0
        self.embeddingSize = 0
        self.hiddenSize = 0
        self.numLayers = 0
        self.numClasses = 0
        self.learningRate = 0
        self.momentum = 0
        self.maxEpoch = 0
        self.trainSize = 0
        self.testSize = 0
        self.sequenceLength = 0
        self.batchSize = 0
        self.initWeight = 0
        self.folds = 2
        self.threshold = 100
        self.learningRateDecay = 0.01
        self.testPer = 0.25
        self.trainPer = 1
        self.saveModel = False


class ModelResults(object):
    def __init__(self):
        self.correct = 0
        self.predPositives = 0
        self.predNegatives = 0
        self.trueNegatives = 0
        self.truePositives = 0
        self.falseNegatives = 0
        self.falsePositive = 0
        self.allPositives = 0
        self.allNegatives = 0
        self.all = self.allTrue = self.allFalse = 0
        self.posRate = self.negRate = 0
        self.precision = self.recall = self.f1Score = self.score = self.accuracy = self.errorRate = 0.0


    def load(self,correct,allNegatives,allPositives,negPred,posPred,falseNeg,falsePos,trueNeg,truePos):
        self.correct = correct
        self.allNegatives = allNegatives
        self.allPositives = allPositives
        self.predNegatives = negPred
        self.predPositives = posPred
        self.falseNegatives = falseNeg
        self.falsePositives = falsePos
        self.trueNegatives = trueNeg
        self.truePositives = truePos
        self.all = allNegatives + allPositives
        self.allTrue = trueNeg + truePos
        self.allFalse = falseNeg + falsePos
        self.posRate = self.negRate = 0
        if self.allPositives != 0:
            self.posRate = self.truePositives / self.allPositives
        if self.allNegatives != 0:
            self.negRate = self.trueNegatives / self.allNegatives

    def calculateScores(self):
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.score = 0.0
        self.accuracy = 0.0
        self.error_rate = 0.0
        self.accuracy = self.allTrue / self.all
        self.errorRate = (self.all - self.allTrue) / self.all
        self.score = self.correct / self.all
        if self.predPositives != 0:
            self.precision = self.truePositives / self.predPositives
        if self.allPositives != 0:
            self.recall = self.truePositives / self.allPositives
        if (self.precision + self.recall) != 0:
            self.f1Score = (2 * self.precision * self.recall / (self.precision + self.recall))
        return True # calculated

def calculateAndPrintScores(outFile,fold,lrStr,momentumStr,maxEpoch, batchSize,results):
    if results.calculateScores() == True:
        outFile.write("cdlc,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,\n" %
                      (fold, lrStr, momentumStr, maxEpoch, batchSize,results.allPositives,
                       results.allNegatives, results.correct, results.predPositives,
                       results.predNegatives, results.truePositives, results.trueNegatives, results.falsePositives,
                       results.falseNegatives,
                       results.precision, results.recall, results.f1Score, results.score, results.accuracy,
                       results.errorRate))
    else:
        outFile.write("Can not calculated")


def testClassifier(classifier, allData, targetsData):
    Tresults = ModelResults()
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0
    TallNegatives = TallPositives = 0

    for i in range(allData.size()[0]):
        x = torch.Tensor(1, allData[i].size()[0])
        x.copy_(allData[i].data)
        x = autograd.Variable(x.cuda())
        pred = classifier.forward(x)
        if targetsData[i] == 0:
            TallNegatives += 1
        else:
            TallPositives += 1

        if pred.data[0][0] < 0.5:
            output = 0
            TnegPred +=  1
        else:
            output = 1
            TposPred +=  1

        print("predicted : %f - target : %d - output : %d" % (pred.data[0][0], targetsData[i],output))

        if output == targetsData[i]:
            Tcorrect +=  1
            if targetsData[i] == 1:
                TtruePos += 1
            if targetsData[i] == 0:
                TtrueNeg +=  1
        else:
            if output == 1:
                TfalsePos += 1
            if output == 0:
                TfalseNeg +=  1


    print("all_pos : %d all_neg : %d " % (TallPositives,TallNegatives))
    print("correct : %d pred_pos : %d pred_neg : %d" % (Tcorrect,TposPred,TnegPred))
    print("true_pos : %d  true_neg : %d" % (TtruePos,TtrueNeg))

    Tresults.load(Tcorrect,TallNegatives,TallPositives,TnegPred,TposPred,TfalseNeg,TfalsePos,TtrueNeg,TtruePos)
    return Tresults

def testClassifier2(classifier, allData, targetsData):
    Tresults = ModelResults()
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0
    TallNegatives = TallPositives = 0

    for i in range(allData.size()[0]):
        x = torch.Tensor(1, allData[i].size()[0])
        x.copy_(allData[i].data)
        x = autograd.Variable(x.cuda())
        pred = classifier.forward(x)
        pr = torch.max(F.softmax(pred), 1)[1]
        if targetsData[i] == 0:
            TallNegatives += 1
        else:
            TallPositives += 1

        if pr.data[0][0] == 0:
            output = 0
            TnegPred +=  1
        else:
            output = 1
            TposPred +=  1

        #print("predicted : %f - target : %d - output : %d" % (pr.data[0][0], targetsData[i],output))

        if output == targetsData[i]:
            Tcorrect +=  1
            if targetsData[i] == 1:
                TtruePos += 1
            if targetsData[i] == 0:
                TtrueNeg +=  1
        else:
            if output == 1:
                TfalsePos += 1
            if output == 0:
                TfalseNeg +=  1


    print("all_pos : %d all_neg : %d " % (TallPositives,TallNegatives))
    print("correct : %d pred_pos : %d pred_neg : %d" % (Tcorrect,TposPred,TnegPred))
    print("true_pos : %d  true_neg : %d" % (TtruePos,TtrueNeg))

    Tresults.load(Tcorrect,TallNegatives,TallPositives,TnegPred,TposPred,TfalseNeg,TfalsePos,TtrueNeg,TtruePos)
    return Tresults


def testClassifierLSTM(Tmodel,TestData,TinpSize):
    Tresults = ModelResults()
    posValues = list(TestData[1].values())
    negValues = list(TestData[0].values())
    TallPositives=len(posValues)
    TallNegatives=len(negValues)

    TestTarget = 1
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0
    for i in range(TallPositives):
        Tcx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thidden = [Thx, Tcx]
        Tinp = [posValues[i].data.clone()]
        pred = Tmodel.forward(Tinp, Thidden, Tinp[0].size()[0])
        if abs(pred.data.round()[0][0]) == TestTarget:
            Tcorrect += 1
            TposPred += 1
            TtruePos += 1
        else:
            TnegPred += 1
            TfalseNeg += 1

        print("target num %d pred:%f pred(round):%f target:%f - correct:%d" % (i,pred.data[0][0], pred.data.round()[0][0], TestTarget, Tcorrect))

    TestTarget = 0
    for i in range(TallNegatives):
        Tcx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thidden = [Thx, Tcx]
        Tinp = [negValues[i].data.clone()]
        pred = Tmodel.forward(Tinp, Thidden, Tinp[0].size()[0])
        if abs(pred.data.round()[0][0]) == TestTarget:
            Tcorrect += 1
            TnegPred += 1
            TtrueNeg += 1
        else:
            TposPred += 1
            TfalsePos += 1

        print("target num %d pred:%f pred(round):%f target:%f - correct:%d" % (i,pred.data[0][0], pred.data.round()[0][0], TestTarget, Tcorrect))

    print("all_pos : %d all_neg : %d " % (TallPositives,TallNegatives))
    print("correct : %d pred_pos : %d pred_neg : %d" % (Tcorrect,TposPred,TnegPred))
    print("true_pos : %d  true_neg : %d" % (TtruePos,TtrueNeg))

    Tresults.load(Tcorrect,TallNegatives,TallPositives,TnegPred,TposPred,TfalseNeg,TfalsePos,TtrueNeg,TtruePos)
    return Tresults



def saveClassifierModel(model,prmStr):
    fnamePth = path + 'classifierModel1000EnTr'+prmStr+".pth"
    fnamePck = path + 'classifierModel1000EnTr' + prmStr + ".pck"
    print('Classifier Model file pth %s' % fnamePth)
    print('Classifier Model file pck %s' % fnamePck)
    torch.save(model.state_dict(), fnamePth)   # Save model file as torch file
    fh = open(fnamePck, 'wb')  # Save model file as pickle
    pickle.dump(model, fh)
    fh.close()

