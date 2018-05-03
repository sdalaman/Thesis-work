
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

import modelresults as mres

class args(object):
    pass
    
class modelResults2(object):
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
        self.precision = self.recall = self.f1_score = self.score = self.accuracy = self.error_rate = 0.0


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
        if self.predPositives != 0 and self.truePositives != 0:
            self.precision = self.truePositives / self.predPositives
            self.recall = self.truePositives / self.allPositives
            self.accuracy = self.allTrue / self.all
            self.f1_score = (2 * self.precision * self.recall / (self.precision + self.recall))
            self.error_rate = (self.all - self.allTrue) / self.all
            self.score = self.correct / self.all
            return True # calculated
        else:
            return False  # not calculated



class BiLingual(nn.Module):
    def __init__(self, vocab_size_pri, vocab_size_sec, embeddingDim):
        super(BiLingual, self).__init__()
        self.embeddings_pri = nn.Embedding(vocab_size_pri, embeddingDim)
        self.embeddings_sec = nn.Embedding(vocab_size_sec, embeddingDim)


    def init_weights(self):
        initrange = 0.01
        init.uniform(self.embeddings_pri.weight, -1 * initrange, initrange)
        init.uniform(self.embeddings_sec.weight, -1 * initrange, initrange)

    def cAdd(self, embeds):
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            tot = autograd.Variable(torch.zeros(embeddingDim).cuda(), requires_grad=True)
            for j in range(sntc_len):
                tot = tot + embeds[i][j]
            ret.append(tot)
        ret = torch.stack(ret, 0)
        return ret

    def forwardPri(self, inputs):
        embeds_pri = self.embeddings_pri(inputs)
        out_pri = self.cAdd(embeds_pri)
        return out_pri

    def forwardSec(self, inputs):
        embeds_sec = self.embeddings_sec(inputs)
        out_sec = self.cAdd(embeds_sec)
        return out_sec

def forwardTanh(embedding,inputs,embedDim):
        embeds = embedding(inputs)
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            tot = torch.zeros(embedDim).cuda()
            for j in range(sntc_len - 1):
                tot = tot + F.tanh(embeds[i][j].data + embeds[i][j + 1].data)
            ret.append(autograd.Variable(tot, requires_grad=True))
        ret = torch.stack(ret, 0)
        return ret


def forwardAdd(embedding, inputs, embedDim):
    embeds = embedding(inputs)
    btch_len = embeds.size()[0]
    sntc_len = embeds.size()[1]
    ret = []
    for i in range(btch_len):
        tot = torch.zeros(embedDim).cuda()
        for j in range(sntc_len - 1):
            tot = tot + embeds[i][j].data
        ret.append(autograd.Variable(tot, requires_grad=True))
    ret = torch.stack(ret, 0)
    return ret


class LSTMClassifier(nn.Module):
    def __init__(self,inputSize,lr_init,momentum,depth):
        super(LSTMClassifier, self).__init__()
        self.inputSize = inputSize
        self.lr_init = lr_init
        self.momentum = momentum
        self.lstm_depth = depth
        self.lstm = nn.LSTMCell(self.inputSize, self.inputSize)
        self.fc = nn.Linear(inputSize,1)
        self.sgmd = nn.Sigmoid().cuda()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr_init, alpha=0.99, eps=1e-08, weight_decay=0, momentum=self.momentum,centered=False)

    def initHidden(self,initrange):
        for ww in self.parameters():
            init.uniform(ww.data, -1 * initrange, initrange)

    def forward(self,inp,hidden,step):
        t1 = torch.zeros(step,inp.size()[0],inp.size()[2]).cuda()
        for i in range(inp.size()[0]):
            for j in range(step):
                t1[j,i] = inp[i][j]
        for j in range(step):
            x = autograd.Variable(t1[j])
            hx,cx = self.lstm(x,hidden)
            hidden = (hx, cx)
        y1 = self.fc(hx)
        yhat = F.sigmoid(y1.clone())
        return yhat,hx,cx


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
    file_sent_list = {}
    for fl in data:
        sentences = []
        for sent in data[fl]:
            x = torch.Tensor(longest_sent)
            newSent = padding(sent, longest_sent)
            for idx in range(longest_sent):
                if vocab.get(newSent[idx]) != None:
                    x[idx] = vocab[newSent[idx]]
                else:
                    x[idx] = 0
            sentences.append(x)
        file_sent_list[fl] = sentences
    return file_sent_list

def getData(data_path,classname,vocab):
    pos_data = {}
    neg_data = {}
    sentences = []
    longest_sent = 0
    path = data_path+'/'+classname+'/positive'
    for file in os.listdir(path):
        sentences = []
        with open(path+"/"+file, 'r') as f:
            for line in f:
                for snt in line.split("."):
                    if not snt == '\n':
                        sentences.append(snt.split())
                        snt_len = len(snt.split())
                        if snt_len >  longest_sent:
                            longest_sent = snt_len
        pos_data[file]= sentences

    path = data_path+'/'+classname+'/negative'
    for file in os.listdir(path):
        sentences = []
        with open(path+"/"+file, 'r') as f:
            for line in f:
                for snt in line.split("."):
                    if not snt == '\n':
                        sentences.append(snt.split())
                        snt_len = len(snt.split())
                        if snt_len >  longest_sent:
                            longest_sent = snt_len
        neg_data[file]=sentences

    pos_mapped = map_data(pos_data, longest_sent, vocab)
    neg_mapped = map_data(neg_data, longest_sent, vocab)
    return pos_mapped, neg_mapped

def adjust_learning_rate(optimizer, epoch,threshold,lr_init,lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (lr_decay_rate ** (epoch // threshold))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def scaleVector(input):
    scaled = torch.mul(torch.add(input, -1 * torch.min(input)), 1 / (torch.max(input) - torch.min(input)))
    return scaled

def trainClassifierLSTM(allDataTrain,allDataTest,targetsTrain,targetsTest,embeddingDim,initWeight,learningRate,momentum,maxEpoch,batchSize,folds,lstmDepth,saveModelStats,results):
    classifier = LSTMClassifier(embeddingDim, learningRate, momentum,lstmDepth).cuda()
    classifier.initHidden(initWeight)
    lrStr = mres.floatToStr("%2.15f", learningRate)
    momentumStr = mres.floatToStr("%2.15f", momentum)
    lossVal = mres.LossValues()
    errors = []
    #results = []
    totWndCount = allDataTrain.size()[0]
    wndBatch = math.floor(totWndCount / batchSize)
    #if totWndCount % batchSize != 0:
    #    wndBatch += 1
    allCnt = 0
    for fold in range(folds):
        #errors = []
        for epoch in range(maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold+1,epoch+1))
            shuffle = np.random.permutation(totWndCount)
            for btcCnt in range(wndBatch):
                index = torch.from_numpy(shuffle[btcCnt*batchSize:(btcCnt+1)*batchSize]).cuda()
                inp = torch.index_select(allDataTrain,0,index)
                targets = autograd.Variable(torch.index_select(targetsTrain,0,index).float(), requires_grad=False)
                cx = autograd.Variable(torch.zeros(batchSize, embeddingDim).cuda())
                hx = autograd.Variable(torch.zeros(batchSize, embeddingDim).cuda())
                hidden = [hx, cx]
                allCnt += 1
                classifier.zero_grad()
                preds,hx,cx = classifier.forward(inp,hidden,lstmDepth)
                loss = ((preds - targets) * (preds - targets)).mean()
                print("fold %d epoch %d lstmDepth %d lr %s mmt %s btcCnt %d - loss %f " %
                          (fold+1,epoch+1,lstmDepth,lrStr,momentumStr,btcCnt+1,loss.data[0]))
                loss.backward()
                classifier.optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if epoch % 50 == 0 and epoch != 0:
                trainresults = testClassifierLSTMNew(classifier,allDataTrain,targetsTrain,embeddingDim,lstmDepth)
                res = "train - lr %s mmt %s maxepoch %d epoch %d lstmDepth %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                lrStr, momentumStr, maxEpoch, epoch+1,lstmDepth, trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate)
                results.append(res)

                testresults = testClassifierLSTMNew(classifier, allDataTest,targetsTest, embeddingDim,lstmDepth)
                res = "test - lr %s mmt %s maxepoch %d epoch %d lstmDepth %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                    lrStr, momentumStr, maxEpoch, epoch+1,lstmDepth, testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate)
                results.append(res)

    trainresults = testClassifierLSTMNew(classifier, allDataTrain,targetsTrain, embeddingDim,lstmDepth)
    res = "train - lr %s mmt %s maxepoch %d epoch %d lstmDepth %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
        lrStr, momentumStr, maxEpoch, maxEpoch ,lstmDepth, trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
        trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate)
    results.append(res)

    testresults = testClassifierLSTMNew(classifier, allDataTest,targetsTest, embeddingDim,lstmDepth)
    res = "test - lr %s mmt %s maxepoch %d epoch %d lstmDepth %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
        lrStr, momentumStr, maxEpoch, maxEpoch ,lstmDepth, testresults.correct, testresults.all,testresults.trueNegatives,testresults.allNegatives,
        testresults.negRate, testresults.truePositives, testresults.allPositives,testresults.posRate)
    results.append(res)

    #for res in results:
    #    print(res)

    if saveModelStats == True:
        lossVal.x = range(allCnt)
        lrStr = mres.floatToStr("%2.15f", learningRate)
        initStr = mres.floatToStr("%2.15f", initWeight)
        fname = "%s%s-lstm-docwindow-cdlc-loss-values-%s-%s-%s-%d-%d-%d.bin" % (path,compositionalMethod,lrStr,momentumStr,initStr,batchSize,maxEpoch,lstmDepth)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()
    return classifier


def calculateAndPrintScores2(outFile,fold,lrStr,momentumStr,maxEpoch,batchSize,lstmDepth,results):
        if results.calculateScores() == True:
            outFile.write("cdlc,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,\n" %
                      (fold, lrStr, momentumStr, maxEpoch, batchSize,lstmDepth,results.allPositives, results.allNegatives, results.correct, results.predPositives,
                       results.predNegatives, results.truePositives, results.trueNegatives, results.falsePositives, results.falseNegatives,
                       results.precision, results.recall, results.f1_score, results.score,results.accuracy,results.error_rate))
        else:
            outFile.write("Can not calculated")

def testClassifierLSTMNew(Tmodel,TestData,TestTargets,TinpSize,lstmDepth):
    Tresults = mres.modelResults()
    TallPositives=TallNegatives=0
    Tcorrect = TnegPred = TposPred = TtrueNeg = TtruePos = TfalseNeg = TfalsePos = 0

    for i in range(TestData.size()[0]):
        Tinp = TestData[i].clone()
        Tinp.resize_(1,Tinp.size()[0],Tinp.size()[1])
        Ttarget = autograd.Variable(TestTargets[i].float(), requires_grad=False)
        Tcx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thx = autograd.Variable(torch.zeros(1, TinpSize).cuda())
        Thidden = [Thx, Tcx]
        Tpred,Thx,Tcx = Tmodel.forward(Tinp, Thidden, lstmDepth)
        pred = Tpred.data.round()[0][0]
        if Ttarget.data[0] == 1:
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

        if pred == Ttarget.data[0]:
            Tcorrect += 1

        print("target num %d pred:%f pred(round):%f target:%d - correct:%d" % (i,Tpred.data[0][0],pred,int(Ttarget.data[0]),Tcorrect))


    print("all_pos : %d all_neg : %d " % (TallPositives,TallNegatives))
    print("correct : %d pred_pos : %d pred_neg : %d" % (Tcorrect,TposPred,TnegPred))
    print("true_pos : %d  true_neg : %d" % (TtruePos,TtrueNeg))

    Tresults.load(Tcorrect,TallNegatives,TallPositives,TnegPred,TposPred,TfalseNeg,TfalsePos,TtrueNeg,TtruePos)
    return Tresults

def prepareSentenceEmbeddings(all,positive,negative,modelEmbeddings,forwardFcn,embedDim):
    allPos = {}
    for fl in positive:
        row = len(positive[fl])
        col = len(positive[fl][0])
        sTensor = torch.zeros(row,col).long().cuda()
        rcnt = 0
        for sentence in positive[fl]:
            sTensor[rcnt] = torch.Tensor(sentence).long().cuda()
            rcnt += 1
        stmp = forwardFcn(modelEmbeddings,autograd.Variable(sTensor),embedDim)
        allPos[fl] = normalizeTensor(stmp)
    all[1] = allPos
    print("pos " + str(datetime.datetime.today()))
    allNeg = {}
    for fl in negative:
        row = len(negative[fl])
        col = len(negative[fl][0])
        sTensor = torch.Tensor(row,col).long().cuda()
        rcnt = 0
        for sentence in negative[fl]:
            sTensor[rcnt] = torch.Tensor(sentence).long().cuda()
            rcnt += 1
        stmp = forwardFcn(modelEmbeddings,autograd.Variable(sTensor),embedDim)
        allNeg[fl] = normalizeTensor(stmp)
    all[0] = allNeg
    print("neg " + str(datetime.datetime.today()))
    return all

def prepareDocWindows(Ttrain,Ttest,lstmDepth,embedDim):
    numberOfWndTrain = 0
    numberOfWndTest = 0
    for i in range(len(Ttrain)):
        for j in Ttrain[i].keys():
            numberOfWndTrain += (len(Ttrain[i][j]) - lstmDepth)

    dataTrain = torch.zeros(numberOfWndTrain,lstmDepth,embedDim).float().cuda()
    targetsTrain = torch.zeros(numberOfWndTrain, 1).long().cuda()

    for i in range(len(Ttest)):
        for j in Ttest[i].keys():
            numberOfWndTest += (len(Ttest[i][j]) - lstmDepth)

    dataTest = torch.zeros(numberOfWndTest,lstmDepth,embedDim).float().cuda()
    targetsTest = torch.zeros(numberOfWndTest, 1).long().cuda()

    cnt = 0
    for i in range(len(Ttrain)):
        for j in Ttrain[i].keys():
            sntcCnt = len(Ttrain[i][j])
            for k in range(sntcCnt - lstmDepth):
                dataTrain[cnt] = Ttrain[i][j].data.narrow(0, k, lstmDepth)
                targetsTrain[cnt] = i
                cnt += 1

    cnt = 0
    for i in range(len(Ttest)):
        for j in Ttest[i].keys():
            sntcCnt = len(Ttest[i][j])
            for k in range(sntcCnt - lstmDepth):
                dataTest[cnt] = Ttest[i][j].data.narrow(0, k,lstmDepth)
                targetsTest[cnt] = i
                cnt += 1

    return dataTrain,dataTest,targetsTrain,targetsTest


def normalizeTensor(tnsr):
    return tnsr
    for i in range(tnsr.size()[0]):
        tnsr[i] = autograd.Variable(tnsr[i].data / torch.norm(tnsr[i].data))
    return tnsr

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

embeddingDim = 64
folds = 1 # 10
testPer = 0.25
trainPer = 1
lrList = [1e-3]
momentumList = [0.5]
maxEpoch = 10 #300
batchSizeList = [150]
initWeightList = [0.0001] #0.0001
lstmDepthList = [10]
path = './'
wrdpath = '../wordvectors/'
compositionalMethod = 'additive'

lrStrMdl = mres.floatToStr("%f", 1e-3)
mmtStrMdl = mres.floatToStr("%f", 0.2)
epochMdl = 100
batchSizeMdl = 100
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'

print('Model file %s' % fullModelNamePth)
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )

modelLoadedPck = BiLingual(len(vocabPri),len(vocabSec),embeddingDim).cuda()
modelLoadedPck.init_weights()
modelLoadedPck= pickle.load( open( fullModelNamePck, "rb" ) )

modelLoaded = modelLoadedPck

fResOut = open(path + compositionalMethod + '-' + 'lstm-docwindow-cdlc-run-results.txt', "w")
ftestRes = open(path + compositionalMethod + '-' + 'lstm-docwindow-cdlc-run.txt', "w")
ftestRes.write("name,fold,lr,momentum,max_epoch,batch_size,lstmDepth,all_positives,all_negatives,correct,predicted_positives,predicted_negatives,"
               "true_positives,true_negatives,false_positives,false_negatives,precision,recall,f1_score,score,accuracy,error_rate\n")

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

results = []

for classname in classes:
    save = False
    load = True
    tanhAdd = True
    fnamePri = path + 'allPri' + '.pck'
    fnameSec = path + 'allSec' + '.pck'
    fnamePriTanh = path + 'allPriTanh' + '.pck'
    fnameSecTanh = path + 'allSecTanh' + '.pck'
    allPri = {}
    allSec = {}

    if tanhAdd == False:
        compositionalMethod = 'additive'
    else:
        compositionalMethod = 'tanh'

    if save == True:
        print(datetime.datetime.today())
        positivePri, negativePri = getData(data_pathPri,classname,vocabPri)
        print("%s loaded %s" % (classname,data_pathPri) )

        allPri = {}
        if tanhAdd == False:
            allPri = prepareSentenceEmbeddings(allPri, positivePri, negativePri,modelLoaded.embeddings_pri,forwardAdd,embeddingDim)
        else:
            allPri = prepareSentenceEmbeddings(allPri,positivePri,negativePri,modelLoaded.embeddings_pri,forwardTanh,embeddingDim)


        print(datetime.datetime.today())

        positiveSec, negativeSec= getData(data_pathSec,classname,vocabSec)
        positiveSecT = {}
        negativeSecT = {}
        tot = math.floor(len(positiveSec)*testPer)
        cnt = 0
        for key in positiveSec:
            positiveSecT[key] = positiveSec[key]
            cnt += 1
            if cnt == tot:
                break
        tot = math.floor(len(negativeSec)*testPer)
        cnt = 0
        for key in negativeSec:
            negativeSecT[key] = negativeSec[key]
            cnt += 1
            if cnt == tot:
                break

        positiveSec = positiveSecT
        negativeSec = negativeSecT

        print("%s loaded %s" % (classname,data_pathSec) )

        allSec = {}
        if tanhAdd == False:
            allSec = prepareSentenceEmbeddings(allSec,positiveSec,negativeSec,modelLoaded.embeddings_sec,forwardAdd,embeddingDim)
        else:
            allSec = prepareSentenceEmbeddings(allSec,positiveSec,negativeSec,modelLoaded.embeddings_sec,forwardTanh,embeddingDim)

        print(datetime.datetime.today())

        positivePri = {}
        negativePri = {}
        positiveSec = {}
        negativeSec = {}
        positiveSecT = {}
        negativeSecT = {}

        if tanhAdd == False:
            fPri = open(fnamePri, 'wb')
            fSec = open(fnameSec, 'wb')  # Save model file as pickle
        else:
            fPri = open(fnamePriTanh, 'wb')
            fSec = open(fnameSecTanh, 'wb')  # Save model file as pickle

        pickle.dump(allPri, fPri)
        pickle.dump(allSec, fSec)
        fPri.close()
        fSec.close()

    if load == True:
        if tanhAdd == False:
            allPri = pickle.load(open(fnamePri, "rb"))
            allSec = pickle.load(open(fnameSec, "rb"))
            print("Train and Test Additive data loaded")

        else:
            allPri = pickle.load(open(fnamePriTanh, "rb"))
            allSec = pickle.load(open(fnameSecTanh, "rb"))
            print("Train and Test Tanh data loaded")

    mx = 0
    for i in range(len(allPri)):
        for j in allPri[i].keys():
            x = allPri[i][j].size()[0]
            if mx < x:
                mx = x
    print("Pri Max Doc Length ( Sentence) : %d" % mx)

    mx = 0
    for i in range(len(allSec)):
        for j in allSec[i].keys():
            x = allSec[i][j].size()[0]
            if mx < x:
               mx = x
    print("Sec Max Doc Length ( Sentence) : %d" % mx)


    results.append("class : " + classname)
    for lstmDepth in lstmDepthList:
        allPriD, allSecD, targetsPriD, targetsSecD = prepareDocWindows(allPri, allSec, lstmDepth, embeddingDim)
        for momentum in momentumList:
            for lr in lrList:
                for batchSize in batchSizeList:
                    for initWeight in initWeightList:
                        classifierTrained=trainClassifierLSTM(allPriD,allSecD,targetsPriD,targetsSecD,embeddingDim,initWeight,lr,momentum,
                                                              maxEpoch,batchSize,folds,lstmDepth,True,results)

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

ftestRes.close()
fResOut.close()
print("End of classifier test")

