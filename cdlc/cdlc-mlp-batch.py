
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
    return model

def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

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

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def scaleVector(input):
    scaled = torch.mul(torch.add(input, -1 * torch.min(input)), 1 / (torch.max(input) - torch.min(input)))
    return scaled

def trainClassifier(allDataTrain,targetDataTrain,allDataTest,targetDataTest,learning_rate,momentum,maxEpoch,saveModel,results):
    classifier = buildClassifierModel(embedding_dim, 1).cuda()
    for param in classifier.parameters():
        init.uniform(param, -1 * 0.0000001, 0.0000001)
    loss_function = nn.BCELoss(size_average=True).cuda()
    optimizer = optim.RMSprop(classifier.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,momentum=momentum, centered=False)
    lossVal = mres.LossValues()
    errors = []
    lrStr = mres.floatToStr("%2.15f",learning_rate)
    momentumStr = mres.floatToStr("%2.15f",momentum)
    epc = 0
    numberOfDoc = math.floor((allDataTrain.size()[0]/batch_size)*batch_size)
    for fold in range(folds):
        for epoch in range(maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold,epoch))
            epc += 1
            inds = torch.range(1, numberOfDoc, batch_size).long()
            shuffle = torch.randperm(inds.size()[0])
            lr=optimizer.param_groups[0]['lr']
            lrStr = mres.floatToStr("%2.15f",lr)
            for i in range(int(numberOfDoc/batch_size)):
                start = inds[shuffle[i]] - 1
                endd = inds[shuffle[i]] + batch_size - 1
                inp = autograd.Variable(allDataTrain[start:endd].data.cuda(), requires_grad=False)
                target = autograd.Variable(torch.Tensor(batch_size).copy_(targetDataTrain[start:endd]).cuda(), requires_grad=False)
                classifier.zero_grad()
                pred = classifier.forward(inp)
                loss = loss_function(pred,target)
                print("fold %d epoch %d lr %s - pred %f target %f loss %f " % (fold,epoch,lrStr,pred.data[0][0],target.data[0], loss.data[0]))
                loss.backward()
                optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if epoch % 50 == 0 and epoch != 0:
                trainresults = mres.testClassifier(classifier,allDataTrain,targetDataTrain)
                res = "train - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                lrStr, momentumStr, maxEpoch, epoch+1,trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate)
                results.append(res)

                testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
                res = "test - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
                    lrStr, momentumStr, maxEpoch, epoch+1,testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate)
                results.append(res)

        trainresults = mres.testClassifier(classifier, allDataTrain, targetDataTrain)
        res = "train - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
            lrStr, momentumStr, maxEpoch, maxEpoch, trainresults.correct, trainresults.all,
            trainresults.trueNegatives, trainresults.allNegatives,
            trainresults.negRate, trainresults.truePositives, trainresults.allPositives, trainresults.posRate)
        results.append(res)

        testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
        res = "test - lr %s mmt %s maxepoch %d epoch %d score %d/%d - trueNegPred/allNeg:%d/%d=%f  truePosPred/allPos:%d/%d=%f" % (
            lrStr, momentumStr, maxEpoch, maxEpoch, testresults.correct, testresults.all,
            testresults.trueNegatives, testresults.allNegatives,
            testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate)
        results.append(res)

    if saveModel == True:
        lossVal.x = range(folds * maxEpoch * int(numberOfDoc/batch_size))
        lrStr = mres.floatToStr("%2.15f",learning_rate)
        fname = "%scdlc-mlp-batch-loss-values-%s-%s-%d.bin" % (path,lrStr,momentumStr,maxEpoch)
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
lrStrMdl = mres.floatToStr("%f" , 1e-3)
mmtStrMdl = mres.floatToStr("%f" , 0.2)
epochMdl = 100
batchSizeMdl = 100
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'
embedding_dim = 64
batch_size = 100
max_epoch = 100 # 300
learning_rate_decay = 0.01
threshold = 100
folds = 20 # 10
test_per = 0.25
train_per = 1
numOfTrCopy=5

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

print('Model file %s' % fullModelNamePth)
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )

modelLoadedPck = BiLingual(len(vocabPri),len(vocabSec),embedding_dim).cuda()
modelLoadedPck.init_weights()
modelLoadedPck= pickle.load( open( fullModelNamePck, "rb" ) )

modelLoaded = modelLoadedPck
results = []

fResOut = open(path + compositionalMethod + '-' + 'cdlc-mlp-batch-run.txt', "w")
ftestRes = open("cdlc-mlp-batch-run.txt", "w")
ftestRes.write("name,fold,lr,momentum,max_epoch,all_positives,all_negatives,correct,predicted_positives,predicted_negatives,"
               "true_positives,true_negatives,precision,recall,f1_score,score,accuracy,error_rate\n")

for classname in classes:
    print(datetime.datetime.today())
    positivePri, negativePri = getData(data_pathPri,classname,vocabPri)
    print("%s loaded %s" % (classname,data_pathPri) )
    all_rawPri = torch.cat([positivePri,negativePri], 0)
    targetsPri = torch.cat([torch.Tensor(positivePri.size()[0]).fill_(1),torch.Tensor(negativePri.size()[0]).fill_(0)],0 )
    allPri = modelLoaded.forwardPri(autograd.Variable(all_rawPri.long().cuda()))

    allPriT=allPri
    targetsPriT=targetsPri
    for cnt in range(numOfTrCopy-1):
        allPri = torch.cat([allPri, allPriT], 0)
        targetsPri = torch.cat([targetsPri, targetsPriT], 0)


    print(datetime.datetime.today())
    all_positivesPri=positivePri.size()[0]*numOfTrCopy
    all_negativesPri=negativePri.size()[0]*numOfTrCopy

    positiveSec, negativeSec= getData(data_pathSec,classname,vocabSec)
    print("%s loaded %s" % (classname,data_pathSec) )
    positiveSec = positiveSec[0:math.floor(positiveSec.size()[0]*test_per),]
    negativeSec = negativeSec[0:math.floor(negativeSec.size()[0]*test_per),]
    all_rawSec = torch.cat([positiveSec,negativeSec], 0)
    targetsSec = torch.cat([torch.Tensor(positiveSec.size()[0]).fill_(1),torch.Tensor(negativeSec.size()[0]).fill_(0)],0 )
    allSec = modelLoaded.forwardSec(autograd.Variable(all_rawSec.long().cuda()))
    all_positivesSec=positiveSec.size()[0]
    all_negativesSec=negativeSec.size()[0]

    momentumlist=[0.3]
    lrlist= [1e-9]

    for momentum in momentumlist:
        for lr in lrlist:
            classifierTrained=trainClassifier(allPri,targetsPri,allSec,targetsSec,lr,momentum,max_epoch, True,results)

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

ftestRes.close()
fResOut.close()
print("End of classifier test")

