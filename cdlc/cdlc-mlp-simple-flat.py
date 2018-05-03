
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
            tot = autograd.Variable(torch.zeros(embedding_dim).cuda(), requires_grad=True)
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

def buildClassifierModel(input_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))
    model.add_module("sigmoid", torch.nn.Sigmoid())
    #model.add_module("lin2",nn.Linear(hidden_size, num_classes))
    return model

#def predict(model, x_val):
#    x = Variable(scaleVector(x_val), requires_grad=False)
#    output = model.forward(x)
#    return output.data.numpy().argmax(axis=1)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_1_size,num_classes,lr,momentum):
        super(MLPClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_1_size = hidden_1_size
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.build_model()

    def build_model(self):
        self.fc1 = nn.Linear(self.input_size, self.num_classes)
#        self.relu = nn.ReLU()
#        self.fc2 = nn.Linear(self.hidden_1_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,momentum=self.momentum, centered=False)

    def forward(self, x):
        out = self.fc1(x)
        #out = self.relu(out)
        #out = self.fc2(out)
        return out

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
    return pos_mapped, neg_mapped,longest_sent

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

def trainClassifier(allDataTrain,targetDataTrain,allDataTest,targetDataTest,prms,results):
    classifier = buildClassifierModel(allDataTrain.size()[1], 1).cuda()
    #classifier = MLPClassifier(embedding_dim,embedding_dim,2,learning_rate,momentum).cuda()
    for param in classifier.parameters():
        init.uniform(param, -1 * prms.initWeight, prms.initWeight)
    loss_function = nn.BCELoss(size_average=True).cuda()
    #optimizer = optim.SGD(classifier.parameters(), lr=prms.learningRate, momentum=prms.momentum)
    optimizer = optim.RMSprop(classifier.parameters(), lr=prms.learningRate, alpha=0.99, eps=1e-08, weight_decay=0,momentum=prms.momentum, centered=False)
    lossVal = mres.LossValues()
    errors = []
    lrStr = mres.floatToStr("%2.15f",prms.learningRate)
    momentumStr = mres.floatToStr("%2.15f",prms.momentum)
    epc = 0
    wndBatch = math.floor(allDataTrain.size()[0] / prms.batchSize)
    if allDataTrain.size()[0] % prms.batchSize != 0:
        wndBatch += 1
    for fold in range(prms.folds):
        for epoch in range(prms.maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold,epoch))
            epc += 1
            shuffle = np.random.permutation(allDataTrain.size()[0])
            lr=optimizer.param_groups[0]['lr']
            lrStr = mres.floatToStr("%2.15f",lr)
            for btcCnt in range(wndBatch):
                index = torch.from_numpy(shuffle[btcCnt * prms.batchSize:(btcCnt + 1) * prms.batchSize]).cuda()
                inp = autograd.Variable(torch.index_select(allDataTrain.data, 0, index))
                target = autograd.Variable(torch.index_select(targetDataTrain.cuda(), 0, index), requires_grad=False)
                classifier.zero_grad()
                pred = classifier.forward(inp)
                loss = loss_function(pred,target)
                #loss = classifier.criterion(pred, target)
                print("fold %d epoch %d lr %s mmt %s- pred %f target %f loss %f " % (fold,epoch,lrStr,momentumStr,pred.data[0][0],target.data[0], loss.data[0]))
                loss.backward()
                optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if epoch % modelPrms.threshold == 0 and epoch != 0:
                trainresults = mres.testClassifier(classifier,allDataTrain,targetDataTrain)
                res = "train - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                lrStr, momentumStr, prms.maxEpoch, epoch+1,trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate)
                results.append(res)

                testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
                res = "test - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                    lrStr, momentumStr, prms.maxEpoch, epoch+1,testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate)
                results.append(res)

    trainresults = mres.testClassifier(classifier, allDataTrain, targetDataTrain)
    res = "train - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
            lrStr, momentumStr, prms.maxEpoch, prms.maxEpoch, trainresults.correct, trainresults.all,
            trainresults.trueNegatives, trainresults.allNegatives,
            trainresults.negRate, trainresults.truePositives, trainresults.allPositives, trainresults.posRate)
    results.append(res)

    testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
    res = "test - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
            lrStr, momentumStr, prms.maxEpoch, prms.maxEpoch, testresults.correct, testresults.all,
            testresults.trueNegatives, testresults.allNegatives,
            testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate)
    results.append(res)

    if prms.saveModel == True:
        lossVal.x = range(prms.folds * prms.maxEpoch * wndBatch)
        lrStr = mres.floatToStr("%2.15f" ,prms.learningRate)
        fname = "%scdlc-mlp-simple-flat-loss-values-%s-%s-%d.bin" % (path,lrStr,momentumStr,prms.maxEpoch)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()
    return classifier


# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

path = './'
wrdpath = '../wordvectors/'
compositionalMethod = 'additive'
lrStrMdl = mres.floatToStr("%f",1e-3)
mmtStrMdl = mres.floatToStr("%f",0.2)
epochMdl = 100
batchSizeMdl = 100
embedding_dimMdl = 64
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'

modelPrms = mres.ModelPrm()

modelPrms.embeddingSize = 64
modelPrms.folds = 2
modelPrms.threshold = 1000
modelPrms.learningRate = 0
modelPrms.learningRateDecay = 0.01
modelPrms.momentum = 0
modelPrms.maxEpoch = 100
modelPrms.batchSize = 0
modelPrms.initWeight = 0.01
modelPrms.testPer = 0.25
modelPrms.trainPer = 1
modelPrms.saveModel = True


#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

print('Model file %s' % fullModelNamePth)
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )

modelLoadedPck = BiLingual(len(vocabPri),len(vocabSec),embedding_dimMdl).cuda()
modelLoadedPck.init_weights()
modelLoadedPck= pickle.load( open( fullModelNamePck, "rb" ) )

modelLoaded = modelLoadedPck
results = []

fResOut = open(path + compositionalMethod + '-' + 'cdlc-mlp-simple-flat-run.txt', "w")
ftestRes = open("cdlc-mlp-simple-flat-run.txt", "w")
ftestRes.write("name,fold,lr,momentum,max_epoch,all_positives,all_negatives,correct,predicted_positives,predicted_negatives,"
               "true_positives,true_negatives,precision,recall,f1_score,score,accuracy,error_rate\n")

load = True

def tensorNorm(tns):
    sum = 0.0
    for i in range(tns.size()[0]):
        for j in range(tns.size()[1]):
            sum += tns[i][j]
    return sum


for classname in classes:
    if load == False:
        print(datetime.datetime.today())
        positivePri, negativePri, maxLenPri = getData(data_pathPri,classname,vocabPri)
        print("%s loaded %s" % (classname,data_pathPri) )
        all_rawPri = torch.cat([positivePri,negativePri], 0)
        targetsPri = torch.cat([torch.Tensor(positivePri.size()[0]).fill_(1),torch.Tensor(negativePri.size()[0]).fill_(0)],0 )
        print(datetime.datetime.today())
        all_positivesPri=positivePri.size()[0]  # 122
        all_negativesPri=negativePri.size()[0]  # 118

        positiveSec, negativeSec, maxLenSec = getData(data_pathSec,classname,vocabSec)
        print("%s loaded %s" % (classname,data_pathSec) )
        positiveSec = positiveSec[0:math.floor(positiveSec.size()[0]*modelPrms.testPer),]
        negativeSec = negativeSec[0:math.floor(negativeSec.size()[0]*modelPrms.testPer),]
        all_rawSec = torch.cat([positiveSec,negativeSec], 0)
        targetsSec = torch.cat([torch.Tensor(positiveSec.size()[0]).fill_(1),torch.Tensor(negativeSec.size()[0]).fill_(0)],0 )
        all_positivesSec=positiveSec.size()[0]  # 27
        all_negativesSec=negativeSec.size()[0]  # 26

        if maxLenPri > maxLenSec:
            maxStncLen = maxLenPri
        else:
            maxStncLen = maxLenSec

        all_rawPriT = all_rawPri.clone()
        all_rawSecT = all_rawSec.clone()
        all_rawPri.resize_(all_rawPri.size()[0],maxStncLen)
        all_rawSec.resize_(all_rawSec.size()[0],maxStncLen)

        for i in range(all_rawPri.size()[0]):
            for j in range(all_rawPri.size()[1]):
                if j < all_rawPriT.size()[1]:
                    all_rawPri[i][j] = all_rawPriT[i][j]
                else:
                    all_rawPri[i][j] = 0

        for i in range(all_rawSec.size()[0]):
            for j in range(all_rawSec.size()[1]):
                if j < all_rawSecT.size()[1]:
                    all_rawSec[i][j] = all_rawSecT[i][j]
                else:
                    all_rawSec[i][j] = 0

        #print("Pri %f %f" % (tensorNorm(all_rawPri),tensorNorm(all_rawPriT)))
        #print("Sec %f %f" % (tensorNorm(all_rawSec),tensorNorm(all_rawSecT)))

        all_rawPriT = []
        all_rawSecT = []


        embeds_pri = modelLoaded.embeddings_pri(autograd.Variable(all_rawPri.long().cuda()))
        embeds_sec = modelLoaded.embeddings_sec(autograd.Variable(all_rawSec.long().cuda()))
        allPri = autograd.Variable(embeds_pri.data.resize_(embeds_pri.size()[0],embeds_pri.size()[1]*embeds_pri.size()[2]))
        allSec = autograd.Variable(embeds_sec.data.resize_(embeds_sec.size()[0],embeds_sec.size()[1]*embeds_sec.size()[2]))
        embeds_pri = []
        embeds_sec = []

        torch.save(allPri, open("allPri-mlp-simple-flat.bin", 'wb'))
        torch.save(allSec, open("allSec-mlp-simple-flat.bin", 'wb'))
        torch.save(targetsPri, open("targetsPri-mlp-simple-flat.bin", 'wb'))
        torch.save(targetsSec, open("targetsSec-mlp-simple-flat.bin", 'wb'))
    else:
        allPri = torch.load(open("allPri-mlp-simple-flat.bin", 'rb'))
        allSec = torch.load(open("allSec-mlp-simple-flat.bin", 'rb'))
        targetsPri = torch.load(open("targetsPri-mlp-simple-flat.bin", 'rb'))
        targetsSec = torch.load(open("targetsSec-mlp-simple-flat.bin", 'rb'))
        print("%s binary files loaded " % (classname) )
        all_positivesPri= 122
        all_negativesPri= 118
        all_positivesSec= 27
        all_negativesSec= 26

    for i in range(allPri.size()[0]):
        allPri[i] = scaleVector(allPri[i].data)

    for i in range(allSec.size()[0]):
        allSec[i] = scaleVector(allSec[i].data)

    momentumList=[0,0.1]
    lrList= [1e-5]

    results.append("class : " + classname)

    for momentum in momentumList:
        for lr in lrList:
            modelPrms.learningRate = lr
            modelPrms.momentum = momentum
            modelPrms.initWeight = 0.000001
            modelPrms.maxEpoch = 100
            modelPrms.batchSize = 10
            classifierTrained=trainClassifier(allPri,targetsPri,allSec,targetsSec,modelPrms,results)

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

ftestRes.close()
fResOut.close()
print("End of classifier test")

