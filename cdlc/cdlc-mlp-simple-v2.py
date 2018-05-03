
# coding: utf-8


import os
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
import sys, getopt

import modelresults as mres

class args(object):
    pass

class modelPrmSet(object):
    def __init__(self,prms):
        self.classname = prms[17]
        self.optimName = prms[0]
        self.lr = prms[1]
        self.momentum = prms[2]
        self.initWeight = prms[3]
        self.batchSize = prms[4]
        self.maxEpoch = prms[5]
        self.maxFold = prms[6]
        self.threshold = prms[7]
        self.hLayerNum = prms[8]
        self.inputDims = prms[9]
        self.hiddenDims = prms[10]
        self.outputDims = prms[11]
        self.learningRateDecay = prms[12]
        self.testPer = prms[13]
        self.trainPer = prms[14]
        self.results = {}
        self.results['train'] = []
        self.results['test'] = []
        self.wordVecsCmpMtd = prms[15]
        self.docVecsCmpMtd = prms[16]

    def outStr(self):
        sOut = "%s,%s,%s,%s,%f,%f,%f,%d,%d,%d,%d,%d,%d,%s,%d" % \
            (self.classname,self.wordVecsCmpMtd,self.docVecsCmpMtd,self.optimName,self.lr,self.momentum,self.initWeight, \
            self.hLayerNum,self.batchSize,self.maxEpoch,self.maxFold,self.threshold, \
            self.inputDims,str(self.hiddenDims),self.outputDims)
        return sOut


class BiLingual(nn.Module):
    def __init__(self, vocab_size_pri, vocab_size_sec, embedding_dim):
        super(BiLingual, self).__init__()
        self.embeddings_pri = nn.Embedding(vocab_size_pri, embedding_dim)
        self.embeddings_sec = nn.Embedding(vocab_size_sec, embedding_dim)

    def init_weights(self):
        initrange = 0.01
        init.uniform(self.embeddings_pri.weight, -1 * initrange, initrange)
        init.uniform(self.embeddings_sec.weight, -1 * initrange, initrange)

    def cAdd(self, embeds):
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            tot = torch.zeros(embeds.size()[2]).cuda()
            for j in range(sntc_len):
                tot = tot + embeds[i][j]
            ret.append(tot/sntc_len)
        ret = torch.stack(ret, 0)
        return ret

    def cAddTanh(self, embeds):
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            tot = torch.zeros(embeds.size()[2]).cuda()
            for j in range(sntc_len-1):
                tot = tot + torch.tanh(embeds[i][j] + embeds[i][j + 1])
            ret.append(tot)
        ret = torch.stack(ret, 0)
        return ret

    def cMinMax(self, embeds):
        btch_len = embeds.size()[0]
        #sntc_len = vectors.size()[1]
        ret = []
        docVecs = torch.zeros(btch_len,2 * embeds.size()[2]).cuda()
        for i in range(btch_len):
            for j in range(embeds.size()[2]):
                tMax = torch.max(embeds[i][:,j])
                tMin = torch.min(embeds[i][:,j])
                docVecs[i,j] = tMin
                docVecs[i,j+embeds.size()[2]] = tMax
        return docVecs

    def forwardPri(self, inputs,cvm):
        embeds_pri = self.embeddings_pri(autograd.Variable(inputs))
        if cvm == "add":
            out_pri = self.cAdd(embeds_pri.data)
        elif cvm == "minmax":
            out_pri = self.cMinMax(embeds_pri.data)
        elif cvm == "addtanh":
            out_pri = self.cAddTanh(embeds_pri.data)
        else:
            out_pri = self.cAdd(embeds_pri.data)
        return out_pri

    def forwardSec(self, inputs,cvm):
        embeds_sec = self.embeddings_sec(autograd.Variable(inputs))
        if cvm == "add":
            out_sec = self.cAdd(embeds_sec.data)
        elif cvm == "minmax":
            out_sec = self.cMinMax(embeds_sec.data)
        elif cvm == "addtanh":
            out_sec = self.cAddTanh(embeds_sec.data)
        else:
            out_sec = self.cAdd(embeds_sec.data)
        return out_sec

def padding(sentence,longest_sent):
    new_sentence = []
    for i in range(longest_sent):
        new_sentence.append('<pad>')
    j = 1
    for i in range((longest_sent - len(sentence) + 1), longest_sent + 1):
        new_sentence[i-1] = sentence[j-1]
        j = j + 1
    return new_sentence


def map_data(data,longest_sent,vocab):
    x = torch.Tensor(len(data),longest_sent)
    for idx in range(len(data)):
        all_words = data[idx].split(" ")
        sample = torch.Tensor(longest_sent)
        all_words = padding(all_words,longest_sent)
        for k in range(len(all_words)):
            if vocab.get(all_words[k]) != None:
                sample[k] = vocab[all_words[k]]
            else:
                sample[k] = 0
        x[idx] = sample
    return x

def getData(data_path,classname,vocab):
    pos_data = []
    neg_data = []
    longest_sent = 0
    path = data_path+'/'+classname+'/positive'
    for file in os.listdir(path):
        with open(path+"/"+file, 'r') as f:
            text = f.read()
            words = text.split(" ")
            if longest_sent < len(words):
                longest_sent = len(words)
            pos_data.append(text)

    path = data_path + '/' + classname + '/negative'
    for file in os.listdir(path):
        with open(path+"/"+file, 'r') as f:
            text = f.read()
            words = text.split(" ")
            if longest_sent < len(words):
                longest_sent = len(words)
            neg_data.append(text)

    pos_mapped = map_data(pos_data, longest_sent, vocab)
    neg_mapped = map_data(neg_data, longest_sent, vocab)
    return pos_mapped, neg_mapped

def adjust_learning_rate(optimizer, epoch,threshold,lr_init,lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (lr_decay_rate ** (epoch // threshold))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def updateWeight(model,learning_rate):
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def scaleTensor(input):
    sc = torch.Tensor(input.size()[0],input.size()[1])
    for i in range(sc.size()[0]):
        sc[i] = torch.mul(torch.add(input[i], -1 * torch.min(input[i])), 1 / (torch.max(input[i]) - torch.min(input[i])))
    return sc

def scaleVector(input):
    scaled = torch.mul(torch.add(input, -1 * torch.min(input)), 1 / (torch.max(input) - torch.min(input)))
    return scaled

def selectOptimizer(model,mdlPrms):
# "SGD","RMSprop","Adadelta","Adagrad","Adam","Adamax","ASGD"
    if mdlPrms.optimName == "SGD":
        if mdlPrms.momentum == 0:
            nesterov = False
        else:
            nesterov = True
        optimizer = torch.optim.SGD(model.parameters(), lr=mdlPrms.lr, momentum=mdlPrms.momentum, dampening=0,
                                    weight_decay=0.01, nesterov=nesterov)
    elif mdlPrms.optimName == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=mdlPrms.lr, alpha=0.99, eps=1e-08, weight_decay=0,momentum=mdlPrms.momentum, centered=False)
    elif mdlPrms.optimName == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=mdlPrms.lr, rho=0.9, eps=1e-06, weight_decay=0)
    elif mdlPrms.optimName == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=mdlPrms.lr, lr_decay=0, weight_decay=0)
    elif mdlPrms.optimName == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=mdlPrms.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif mdlPrms.optimName == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=mdlPrms.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif mdlPrms.optimName == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=mdlPrms.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    else:  # default optmizer SGD
        if mdlPrms.momentum == 0:
            nesterov = False
        else:
            nesterov = True
        optimizer = torch.optim.SGD(model.parameters(), lr=mdlPrms.lr, momentum=mdlPrms.momentum, dampening=0,
                                    weight_decay=0, nesterov=nesterov)
    return optimizer

class Net(torch.nn.Module):
    def __init__(self, mdlPrms):
        super(Net, self).__init__()
        self.numOfHiddenLayers=mdlPrms.hLayerNum
        hNum=0
        self.hidden1 = torch.nn.Linear(mdlPrms.inputDims, mdlPrms.hiddenDims[hNum])   # hidden layer
        if self.numOfHiddenLayers >= 2:
            hNum += 1
            self.hidden2 = torch.nn.Linear(mdlPrms.hiddenDims[hNum-1],mdlPrms.hiddenDims[hNum])  # hidden layer
        if self.numOfHiddenLayers >= 3:
            hNum += 1
            self.hidden3 = torch.nn.Linear(mdlPrms.hiddenDims[hNum-1],mdlPrms.hiddenDims[hNum])  # hidden layer
        self.out = torch.nn.Linear(mdlPrms.hiddenDims[hNum], mdlPrms.outputDims)   # output layer

    def forward(self, x):
        x = self.hidden1(x)
        if self.numOfHiddenLayers >= 2:
            x = self.hidden2(x)
        if self.numOfHiddenLayers >= 3:
            x = self.hidden3(x)
        x = self.out(x)
        return x

class Net2(torch.nn.Module):
    def __init__(self, mdlPrms):
        super(Net2, self).__init__()
        self.out1 = torch.nn.Linear(mdlPrms.inputDims, mdlPrms.outputDims)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.out1(x)
        x = self.sig(x)
        return x

def trainClassifierNew(classifier,optimizer,allDataTrain,targetDataTrain,allDataTest,targetDataTest,mdlPrms,saveModel):
    #loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    loss_func = torch.nn.BCELoss(size_average=True).cuda()
    for param in classifier.parameters():
        init.uniform(param, -1 * 0, mdlPrms.initWeight)
    lossVal = mres.LossValues()
    trainStats = mres.ModelStats()
    testStats = mres.ModelStats()
    errors = []
    lrStr = mres.floatToStr("%2.15f",mdlPrms.lr)
    initStr = mres.floatToStr("%2.15f", mdlPrms.initWeight)
    momentumStr = mres.floatToStr("%2.15f",mdlPrms.momentum)
    epc = 0
    reportFlag = False
    wndBatch = math.floor(allDataTrain.size()[0] / mdlPrms.batchSize)
    if allDataTrain.size()[0] % mdlPrms.batchSize != 0:
        wndBatch += 1
    for fold in range(mdlPrms.maxFold):
        for epoch in range(mdlPrms.maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold+1,epoch+1))
            epc += 1
            shuffle = np.random.permutation(allDataTrain.size()[0])
            for btcCnt in range(wndBatch):
                index = torch.from_numpy(shuffle[btcCnt * mdlPrms.batchSize:(btcCnt + 1) * mdlPrms.batchSize]).cuda()
                inp = torch.index_select(allDataTrain.data, 0, index)
                inp = autograd.Variable(inp)
                #target = autograd.Variable(torch.index_select(targetDataTrain.long().cuda(), 0, index))
                target = autograd.Variable(torch.index_select(targetDataTrain.cuda(), 0, index))
                optimizer.zero_grad()
                pred = classifier(inp)
                loss = loss_func(pred, target)
                #print("%s hidden %d opt %s fold %d epoch %d lr %s mmt %s init %s - pred %f/%f target %f loss %f " % (classname,mdlPrms.hLayerNum,mdlPrms.optimName,fold,epoch,lrStr,momentumStr,initStr,pred.data[0][0],pred.data[0][1],target.data[0], loss.data[0]))
                print("class %s opt %s fold %d epoch %d lr %s mmt %s init %s - pred %f target %f loss %f " % (classname,mdlPrms.optimName,fold,epoch,lrStr,momentumStr,initStr,pred.data[0][0],target.data[0], loss.data[0]))
                loss.backward()
                optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if (epoch+1) % mdlPrms.threshold == 0 and epoch != 0 and reportFlag == True:
                trainresults = mres.testClassifier2(classifier,allDataTrain,targetDataTrain)
                trainresults.calculateScores()
                res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr, initStr,mdlPrms.maxEpoch, epoch+1,trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                    trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate,
                    trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score, trainresults.accuracy, trainresults.errorRate )
                mdlPrms.results['train'].append(res)
                trainStats.addStat(fold+1,epoch+1,trainresults)

                testresults = mres.testClassifier2(classifier, allDataTest, targetDataTest)
                testresults.calculateScores()
                res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr,initStr, mdlPrms.maxEpoch, epoch+1,testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
                    testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
                    testresults.accuracy, testresults.errorRate)
                mdlPrms.results['test'].append(res)
                testStats.addStat(fold+1, epoch + 1,testresults)


        trainresults = mres.testClassifier2(classifier, allDataTrain, targetDataTrain)
        trainresults.calculateScores()
        res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr, initStr,mdlPrms.maxEpoch, mdlPrms.maxEpoch, trainresults.correct, trainresults.all,
            trainresults.trueNegatives, trainresults.allNegatives,
            trainresults.negRate, trainresults.truePositives, trainresults.allPositives, trainresults.posRate,
            trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score,
            trainresults.accuracy, trainresults.errorRate)
        mdlPrms.results['train'].append(res)
        trainStats.addStat(fold+1, mdlPrms.maxEpoch,trainresults)

        testresults = mres.testClassifier2(classifier, allDataTest, targetDataTest)
        testresults.calculateScores()
        res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr,initStr, mdlPrms.maxEpoch, mdlPrms.maxEpoch, testresults.correct, testresults.all,
            testresults.trueNegatives, testresults.allNegatives,
            testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
            testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
            testresults.accuracy, testresults.errorRate)
        mdlPrms.results['test'].append(res)
        testStats.addStat(fold+1, mdlPrms.maxEpoch, testresults)

    if saveModel == True:
        lossVal.x = range(mdlPrms.maxFold * mdlPrms.maxEpoch * math.floor(allDataTrain.size()[0] * mdlPrms.trainPer))
        lrStr = mres.floatToStr("%2.15f" ,mdlPrms.lr)
        fname = "%scdlc-%s-%s-%s-mlp-simple-loss-values-%s-%s-%s-%d-%d-%d.bin" % (path,classname,modelPrms.wordVecsCmpMtd,modelPrms.docVecsCmpMtd,mdlPrms.optimName,lrStr,momentumStr,mdlPrms.maxEpoch,mdlPrms.maxFold,mdlPrms.hLayerNum)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()
        mdlStats = {"train" : trainStats,"test":testStats}
        fname = "%scdlc-%s-%s-%s-mlp-simple-stat-values-%s-%s-%s-%d-%d-%d.bin" % (path,classname,modelPrms.wordVecsCmpMtd,modelPrms.docVecsCmpMtd,mdlPrms.optimName,lrStr,momentumStr,mdlPrms.maxEpoch,mdlPrms.maxFold,mdlPrms.hLayerNum)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(mdlStats, fh)
        fh.close()
    return classifier

classinp = ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"h",["class="])
except getopt.GetoptError:
    print ('prog.py --class <classname>')
    sys.exit(2)
for opt,arg in opts:
    if opt == '-h':
        print('prog.py -class <classname>')
        sys.exit()
    elif opt in ("--class"):
        classinp = arg

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

path = './'
wrdpath = '../wordvectors/1000/'
cmpMethodWord2Vec = 'additive' #'tanh'
lrStrMdl = mres.floatToStr("%f",1e-3)
mmtStrMdl = mres.floatToStr("%f",0.2)
epochMdl = 500
batchSizeMdl = 100
embeddingDimMdl = 64

prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath + cmpMethodWord2Vec + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath + cmpMethodWord2Vec + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath + cmpMethodWord2Vec + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + cmpMethodWord2Vec + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

print('Model file %s' % fullModelNamePth)
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )

modelLoadedPck = BiLingual(len(vocabPri),len(vocabSec),embeddingDimMdl).cuda()
modelLoadedPck.init_weights()
modelLoadedPck= pickle.load( open( fullModelNamePck, "rb" ) )

modelLoaded = modelLoadedPck

load = True
test_per = 0.25
train_per = 1
allPri = {}
allSec = {}

if classinp != "":
    classes = []
    classes.append(classinp)

for classname in classes:
    if load == False:
        print(datetime.datetime.today())
        positivePri, negativePri = getData(data_pathPri,classname,vocabPri)
        positiveSec, negativeSec = getData(data_pathSec, classname, vocabSec)

        print("%s loaded %s" % (classname,data_pathPri) )
        all_rawPri = torch.cat([positivePri,negativePri], 0)
        targetsPri = torch.cat([torch.Tensor(positivePri.size()[0]).fill_(1),torch.Tensor(negativePri.size()[0]).fill_(0)],0 )
        #allPri["add"] = autograd.Variable(modelLoaded.forwardPri(all_rawPri.long().cuda(),"add"))
        allPri["minmax"] = autograd.Variable(modelLoaded.forwardPri(all_rawPri.long().cuda(),"minmax"))
        #allPri["tanh"] = autograd.Variable(modelLoaded.forwardPri(all_rawPri.long().cuda(),"addtanh"))
        print(datetime.datetime.today())
        #all_positivesPri=positivePri.size()[0]  # 122
        #all_negativesPri=negativePri.size()[0]  # 118

        print("%s loaded %s" % (classname,data_pathSec) )
        positiveSec = positiveSec[0:math.floor(positiveSec.size()[0]*test_per),]
        negativeSec = negativeSec[0:math.floor(negativeSec.size()[0]*test_per),]
        all_rawSec = torch.cat([positiveSec,negativeSec], 0)
        targetsSec = torch.cat([torch.Tensor(positiveSec.size()[0]).fill_(1),torch.Tensor(negativeSec.size()[0]).fill_(0)],0 )
        #allSec["add"] = autograd.Variable(modelLoaded.forwardSec(all_rawSec.long().cuda(),"add"))
        allSec["minmax"] = autograd.Variable(modelLoaded.forwardSec(all_rawSec.long().cuda(),"minmax"))
        #allSec["tanh"] = autograd.Variable(modelLoaded.forwardSec(all_rawSec.long().cuda(),"addtanh"))
        #all_positivesSec=positiveSec.size()[0]  # 27
        #all_negativesSec=negativeSec.size()[0]  # 26

        #pickle.dump(allPri["add"], open(cmpMethodWord2Vec + "-model-allPri-add-mlp-simple.pck", 'wb'))
        #pickle.dump(allSec["add"], open(cmpMethodWord2Vec + "-model-allSec-add-mlp-simple.pck", 'wb'))
        pickle.dump(allPri["minmax"], open(classname + "-" + cmpMethodWord2Vec + "-model-allPri-minmax-mlp-simple.pck", 'wb'))
        pickle.dump(allSec["minmax"], open(classname + "-" + cmpMethodWord2Vec + "-model-allSec-minmax-mlp-simple.pck", 'wb'))
        #pickle.dump(allPri["tanh"], open(cmpMethodWord2Vec + "-model-allPri-tanh-mlp-simple.pck", 'wb'))
        #pickle.dump(allSec["tanh"], open(cmpMethodWord2Vec + "-model-allSec-tanh-mlp-simple.pck", 'wb'))
        pickle.dump(targetsPri, open(classname + "-" + cmpMethodWord2Vec + "-model-targetsPri-mlp-simple.pck", 'wb'))
        pickle.dump(targetsSec, open(classname + "-" + cmpMethodWord2Vec + "-model-targetsSec-mlp-simple.pck", 'wb'))

    else:
        allPri = {"additive":{},"tanh":{}}
        allSec = {"additive":{},"tanh":{}}
        cmpMethodWord2Vec = "additive"
        #allPri[cmpMethodWord2Vec]["add"] = pickle.load(open(cmpMethodWord2Vec + "-model-allPri-add-mlp-simple.pck", 'rb'))
        #allSec[cmpMethodWord2Vec]["add"] = pickle.load(open(cmpMethodWord2Vec + "-model-allSec-add-mlp-simple.pck", 'rb'))
        allPri[cmpMethodWord2Vec]["minmax"] = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-allPri-minmax-mlp-simple.pck", 'rb'))
        allSec[cmpMethodWord2Vec]["minmax"] = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-allSec-minmax-mlp-simple.pck", 'rb'))
        #allPri[cmpMethodWord2Vec]["tanh"] = pickle.load(open(cmpMethodWord2Vec + "-model-allPri-tanh-mlp-simple.pck", 'rb'))
        #allSec[cmpMethodWord2Vec]["tanh"] = pickle.load(open(cmpMethodWord2Vec + "-model-allSec-tanh-mlp-simple.pck", 'rb'))
        #allPri[cmpMethodWord2Vec]["add"] = autograd.Variable(scaleTensor(allPri[cmpMethodWord2Vec]["add"].data).cuda())
        #allSec[cmpMethodWord2Vec]["add"] = autograd.Variable(scaleTensor(allSec[cmpMethodWord2Vec]["add"].data).cuda())

        cmpMethodWord2Vec = "tanh"
        #allPri[cmpMethodWord2Vec]["add"] = pickle.load(open(cmpMethodWord2Vec + "-model-allPri-add-mlp-simple.pck", 'rb'))
        #allSec[cmpMethodWord2Vec]["add"] = pickle.load(open(cmpMethodWord2Vec + "-model-allSec-add-mlp-simple.pck", 'rb'))
        allPri[cmpMethodWord2Vec]["minmax"] = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-allPri-minmax-mlp-simple.pck", 'rb'))
        allSec[cmpMethodWord2Vec]["minmax"] = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-allSec-minmax-mlp-simple.pck", 'rb'))
        #allPri[cmpMethodWord2Vec]["tanh"] = pickle.load(open(cmpMethodWord2Vec + "-model-allPri-tanh-mlp-simple.pck", 'rb'))
        #allSec[cmpMethodWord2Vec]["tanh"] = pickle.load(open(cmpMethodWord2Vec + "-model-allSec-tanh-mlp-simple.pck", 'rb'))
        #allPri[cmpMethodWord2Vec]["add"] = autograd.Variable(scaleTensor(allPri[cmpMethodWord2Vec]["add"].data).cuda())
        #allSec[cmpMethodWord2Vec]["add"] = autograd.Variable(scaleTensor(allSec[cmpMethodWord2Vec]["add"].data).cuda())

        targetsPri = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-targetsPri-mlp-simple.pck", 'rb'))
        targetsSec = pickle.load(open(classname + "-" + cmpMethodWord2Vec + "-model-targetsSec-mlp-simple.pck", 'rb'))
        print("%s Pickle files loaded " % (classname) )


        #all_positivesPri= torch.sum(targetsPri)
        #all_negativesPri= targetsPri.size()[0]  - torch.sum(targetsPri)
        #all_positivesSec= torch.sum(targetsSec)
        #all_negativesSec= targetsSec.size()[0]  - torch.sum(targetsSec)


    prmList = []
    # format : optimizer name, learning rate,momentum,init weight,batch size,max epoch,max fold,
    # threshold,num of hidden layers,input dim,hidden dimensions,output dim,learning rate decay,test pecentage,train percentage,word2vec composition method,
    # doc composition method
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,2,64,[128,128],2,0.01,test_per,train_per,"additive","add",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,2,64,[128,128],2,0.01,test_per,train_per,"tanh","add",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,2,64,[128,128],2,0.01,test_per,train_per,"additive","tanh",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,2,64,[128,128],2,0.01,test_per,train_per,"tanh","tanh",classname])

    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,300,10,1000,2,128,[128,128],1,0.01,test_per,train_per,"additive","minmax",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,300,10,1000,2,128,[128,128],1,0.01,test_per,train_per,"tanh","minmax",classname])

    prmList.append(["RMSprop", 0.0001, 0.5, 0.00001,20,300,10,1000,2,128,[128,128],1,0.01,test_per,train_per,"additive","minmax",classname])

    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,64,[128,128,128],2,0.01,test_per,train_per,"additive","add",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,64,[128,128,128],2,0.01,test_per,train_per,"tanh","add",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,64,[128,128,128],2,0.01,test_per,train_per,"additive","tanh",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,64,[128,128,128],2,0.01,test_per,train_per,"tanh","tanh",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,128,[128,128,128],2,0.01,test_per,train_per,"additive","minmax",classname])
    #prmList.append(["RMSprop", 0.001, 0.5, 0.000001,1,5000,1,100,3,128,[128,128,128],2,0.01,test_per,train_per,"tanh","minmax",classname])

    fResOut = open(path + classname + '-cdlc' + '-mlp-simple-run.txt',"w")

    for prm in prmList:
        modelPrms = modelPrmSet(prm)
        modelPrms.results['train'].append("class : " + classname)
        modelPrms.results['test'].append("class : " + classname)
        modelPrms.results['train'].append("optimizer : " + modelPrms.optimName)
        modelPrms.results['test'].append("optimizer : " + modelPrms.optimName)
        classifier = Net2(modelPrms).cuda()  # define the network
        print(classifier)  # net architecture
        optimizer = selectOptimizer(classifier, modelPrms)

        classifierTrained = trainClassifierNew(classifier,optimizer,allPri[modelPrms.wordVecsCmpMtd][modelPrms.docVecsCmpMtd],
                                               targetsPri, allSec[modelPrms.wordVecsCmpMtd][modelPrms.docVecsCmpMtd],targetsSec,modelPrms,True)

        fResOut.write(modelPrms.outStr())
        fResOut.write("\ntrain\n")
        for res in modelPrms.results['train']:
            print(res)
            fResOut.write(res)
            fResOut.write("\n")

        fResOut.write("test\n")
        for res in modelPrms.results['test']:
            print(res)
            fResOut.write(res)
            fResOut.write("\n")
        fResOut.flush()

    fResOut.close()

print("End of classifier test")

