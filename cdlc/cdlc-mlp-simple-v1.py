
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
            tot = torch.zeros(embedding_dim).cuda()
            for j in range(sntc_len):
                tot = tot + embeds[i][j]
            ret.append(tot/sntc_len)
        ret = torch.stack(ret, 0)
        return ret

    def cMinMax(self, vectors):
        btch_len = vectors.size()[0]
        #sntc_len = vectors.size()[1]
        ret = []
        docVecs = torch.zeros(btch_len,2 * embedding_dim).cuda()
        for i in range(btch_len):
            for j in range(embedding_dim):
                tMax = torch.max(vectors[i][:,j])
                tMin = torch.min(vectors[i][:,j])
                docVecs[i,j] = tMin
                docVecs[i,j+embedding_dim] = tMax
        return docVecs

    def forwardPri(self, inputs,cvm):
        embeds_pri = self.embeddings_pri(autograd.Variable(inputs))
        if cvm == "add":
            out_pri = self.cAdd(embeds_pri.data)
        elif cvm == "minmax":
            out_pri = self.cMinMax(embeds_pri.data)
        else:
            out_pri = self.cAdd(embeds_pri.data)
        return out_pri

    def forwardSec(self, inputs,cvm):
        embeds_sec = self.embeddings_sec(autograd.Variable(inputs))
        if cvm == "add":
            out_sec = self.cAdd(embeds_sec.data)
        elif cvm == "minmax":
            out_sec = self.cMinMax(embeds_sec.data)
        else:
            out_sec = self.cAdd(embeds_sec.data)
        return out_sec

def buildClassifierModel(input_dim,hidden_dim, output_dim):
    model = torch.nn.Sequential()
    model.add_module("linear1", torch.nn.Linear(input_dim, hidden_dim, bias=True))
    model.add_module("tanh1", torch.nn.Tanh())
    model.add_module("linear2", torch.nn.Linear(hidden_dim, hidden_dim, bias=True))
    model.add_module("tanh2", torch.nn.Tanh())
    model.add_module("linear3", torch.nn.Linear(hidden_dim, output_dim, bias=True))
    model.add_module("sigmoid", torch.nn.Sigmoid())
    #model.add_module("ReLU", torch.nn.ReLU())
    #model.add_module("lin2",nn.Linear(hidden_size, num_classes))
    return model

def predict(model, x_val):
    #x = Variable(scaleVector(x_val), requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)


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
        self.fc1 = nn.Linear(self.input_size, self.hidden_1_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_1_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.BCELoss()
        #self.optimizer = optim.SGD(self.parameters(), lr=self.lr,momentum = self.momentum)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,momentum=self.momentum, centered=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
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

def selectOptimizer(model,optimName,learning_rate,momentum):
# "SGD","RMSprop","Adadelta","Adagrad","Adam","Adamax","ASGD"
    if optimName == "SGD":
        if momentum == 0:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0,weight_decay=0, nesterov=False)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate, momentum=momentum, dampening=0, weight_decay=0, nesterov=True)
    if optimName == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,momentum=momentum, centered=False)
    if optimName == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
    if optimName == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0)
    if optimName == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if optimName == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if optimName == "ASGD":
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    return optimizer


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
        return x

def trainClassifierNew(allDataTrain,targetDataTrain,allDataTest,targetDataTest,learning_rate,momentum,maxEpoch,init_weight,saveModel,optimName,results):
    classifier = Net(n_feature=128, n_hidden=128, n_output=2).cuda()  # define the network
    print(classifier)  # net architecture
    #optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
    optimizer = selectOptimizer(classifier,optimName,learning_rate,momentum)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    lossVal = mres.LossValues()
    trainStats = mres.ModelStats()
    testStats = mres.ModelStats()
    errors = []
    lrStr = mres.floatToStr("%2.15f",learning_rate)
    initStr = mres.floatToStr("%2.15f", init_weight)
    momentumStr = mres.floatToStr("%2.15f",momentum)
    epc = 0
    wndBatch = math.floor(allDataTrain.size()[0] / batchSize)
    if allDataTrain.size()[0] % batchSize != 0:
        wndBatch += 1
    for fold in range(folds):
        for epoch in range(maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold,epoch))
            epc += 1
            shuffle = np.random.permutation(allDataTrain.size()[0])
            lr=optimizer.param_groups[0]['lr']
            for btcCnt in range(wndBatch):
                index = torch.from_numpy(shuffle[btcCnt * batchSize:(btcCnt + 1) * batchSize]).cuda()
                #inp = scaleVector(torch.index_select(allDataTrain.data, 0, index))
                inp = torch.index_select(allDataTrain.data, 0, index)
                inp = autograd.Variable(inp)
                target = autograd.Variable(torch.index_select(targetDataTrain.long().cuda(), 0, index))
                optimizer.zero_grad()
                pred = classifier(inp)
                loss = loss_func(pred, target)
                print("opt %s fold %d epoch %d lr %s mmt %s init %s - pred %f/%f target %f loss %f " % (optimName,fold,epoch,lrStr,momentumStr,initStr,pred.data[0][0],pred.data[0][1],target.data[0], loss.data[0]))
                loss.backward()
                optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if (epoch+1) % threshold == 0 and epoch != 0:
                trainresults = mres.testClassifier2(classifier,allDataTrain,targetDataTrain)
                trainresults.calculateScores()
                res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr, initStr,maxEpoch, epoch+1,trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                    trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate,
                    trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score, trainresults.accuracy, trainresults.errorRate )
                results['train'].append(res)
                trainStats.addStat(fold,epoch+1,trainresults)

                testresults = mres.testClassifier2(classifier, allDataTest, targetDataTest)
                testresults.calculateScores()
                res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr,initStr, maxEpoch, epoch+1,testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
                    testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
                    testresults.accuracy, testresults.errorRate)
                results['test'].append(res)
                testStats.addStat(fold, epoch + 1, testresults)


        trainresults = mres.testClassifier2(classifier, allDataTrain, targetDataTrain)
        trainresults.calculateScores()
        res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr, initStr,maxEpoch, maxEpoch, trainresults.correct, trainresults.all,
            trainresults.trueNegatives, trainresults.allNegatives,
            trainresults.negRate, trainresults.truePositives, trainresults.allPositives, trainresults.posRate,
            trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score,
            trainresults.accuracy, trainresults.errorRate)
        results['train'].append(res)
        trainStats.addStat(fold, maxEpoch, trainresults)

        testresults = mres.testClassifier2(classifier, allDataTest, targetDataTest)
        testresults.calculateScores()
        res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr,initStr, maxEpoch, maxEpoch, testresults.correct, testresults.all,
            testresults.trueNegatives, testresults.allNegatives,
            testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
            testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
            testresults.accuracy, testresults.errorRate)
        results['test'].append(res)
        testStats.addStat(fold, maxEpoch, testresults)

    if saveModel == True:
        lossVal.x = range(folds * maxEpoch * math.floor(allDataTrain.size()[0] * train_per))
        lrStr = mres.floatToStr("%2.15f" ,learning_rate)
        fname = "%scdlc-%s-mlp-simple-minmax-loss-values-%s-%s-%s-%s-%d.bin" % (path,classname,optimName,lrStr,momentumStr,initStr,maxEpoch)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()
        mdlStats = {"train" : trainStats,"test":testStats}
        fname = "%scdlc-%s-mlp-simple-minmax-4-hidden-stat-values-%s-%s-%s-%s-%d.bin" % (path,classname,optimName,lrStr,momentumStr,initStr,maxEpoch)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(mdlStats, fh)
        fh.close()
    return classifier



def trainClassifier(allDataTrain,targetDataTrain,allDataTest,targetDataTest,learning_rate,momentum,maxEpoch,init_weight,saveModel,optimName,results):
    classifier = buildClassifierModel(embedding_dim,128,128,1).cuda()
    #classifier = MLPClassifier(embedding_dim,hidden_dim,2,learning_rate,momentum).cuda()
    for param in classifier.parameters():
        init.uniform(param, -1 * init_weight, init_weight)
    loss_function = nn.BCELoss(size_average=True).cuda()
    #loss_function = nn.L1Loss().cuda()
    optimizer = selectOptimizer(classifier,optimName,learning_rate,momentum)
    lossVal = mres.LossValues()
    errors = []
    lrStr = mres.floatToStr("%2.15f",learning_rate)
    initStr = mres.floatToStr("%2.15f", init_weight)
    momentumStr = mres.floatToStr("%2.15f",momentum)
    epc = 0
    wndBatch = math.floor(allDataTrain.size()[0] / batchSize)
    if allDataTrain.size()[0] % batchSize != 0:
        wndBatch += 1
    for fold in range(folds):
        for epoch in range(maxEpoch):
            print("class :  %s fold %d epoch %d" % (classname,fold,epoch))
            epc += 1
            shuffle = np.random.permutation(allDataTrain.size()[0])
            lr=optimizer.param_groups[0]['lr']
            #lrStr = mres.floatToStr("%2.15f",lr)
            #for i in range(math.floor(allDataTrain.size()[0] * train_per)):
            for btcCnt in range(wndBatch):
                index = torch.from_numpy(shuffle[btcCnt * batchSize:(btcCnt + 1) * batchSize]).cuda()
                #inp = scaleVector(torch.index_select(allDataTrain.data, 0, index))
                inp = torch.index_select(allDataTrain.data, 0, index)
                inp = autograd.Variable(inp)
                target = autograd.Variable(torch.index_select(targetDataTrain.cuda(), 0, index))
                #inp = scaleVector(allDataTrain[shuffle[i]].data)
                #inp = autograd.Variable(inp.resize_(1,embedding_dim).cuda(), requires_grad=False)
                #target = autograd.Variable(torch.Tensor(1).fill_(targetDataTrain[shuffle[i]]).long().cuda(), requires_grad=False)
                optimizer.zero_grad()
                pred = classifier.forward(inp)
                #loss = -(target * torch.log(pred) + (1 - target)*torch.log(1 - pred))
                loss = loss_function(pred, target)
                #loss = classifier.criterion(pred, target)
                pr = torch.max(F.softmax(pred), 1)[1]
                print("opt %s fold %d epoch %d lr %s mmt %s init %s - pred %f target %f loss %f " % (optimName,fold,epoch,lrStr,momentumStr,initStr,pred.data[0][0],target.data[0], loss.data[0]))
                loss.backward()
                #nn.utils.clip_grad_norm(classifier.parameters(),0.001)
                optimizer.step()
                errors.append(loss.data[0])
                lossVal.y.append(loss.data[0])
                mean = torch.mean(torch.Tensor(errors))
                lossVal.mean.append(mean)

            if (epoch+1) % threshold == 0 and epoch != 0:
                trainresults = mres.testClassifier(classifier,allDataTrain,targetDataTrain)
                trainresults.calculateScores()
                res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr, initStr,maxEpoch, epoch+1,trainresults.correct, trainresults.all,trainresults.trueNegatives,trainresults.allNegatives,
                    trainresults.negRate, trainresults.truePositives, trainresults.allPositives,trainresults.posRate,
                    trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score, trainresults.accuracy, trainresults.errorRate )
                results['train'].append(res)

                testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
                testresults.calculateScores()
                res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
                    fold,lrStr, momentumStr,initStr, maxEpoch, epoch+1,testresults.correct, testresults.all,
                    testresults.trueNegatives, testresults.allNegatives,
                    testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
                    testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
                    testresults.accuracy, testresults.errorRate)
                results['test'].append(res)


        trainresults = mres.testClassifier(classifier, allDataTrain, targetDataTrain)
        trainresults.calculateScores()
        res = "train - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr, initStr,maxEpoch, maxEpoch, trainresults.correct, trainresults.all,
            trainresults.trueNegatives, trainresults.allNegatives,
            trainresults.negRate, trainresults.truePositives, trainresults.allPositives, trainresults.posRate,
            trainresults.precision, trainresults.recall, trainresults.f1Score, trainresults.score,
            trainresults.accuracy, trainresults.errorRate)
        results['train'].append(res)

        testresults = mres.testClassifier(classifier, allDataTest, targetDataTest)
        testresults.calculateScores()
        res = "test - fold:%d,lr:%s,mmt:%s,init:%s,maxepoch:%d,epoch:%d,score:%d/%d,trueNegPred/allNeg:%d/%d=%f,truePosPred/allPos:%d/%d=%f,precision:%f,recall:%f,f1:%f,score:%f,accuracy:%f,error_rate:%f" % (
            fold, lrStr, momentumStr,initStr, maxEpoch, maxEpoch, testresults.correct, testresults.all,
            testresults.trueNegatives, testresults.allNegatives,
            testresults.negRate, testresults.truePositives, testresults.allPositives, testresults.posRate,
            testresults.precision, testresults.recall, testresults.f1Score, testresults.score,
            testresults.accuracy, testresults.errorRate)
        results['test'].append(res)

    if saveModel == True:
        lossVal.x = range(folds * maxEpoch * math.floor(allDataTrain.size()[0] * train_per))
        lrStr = mres.floatToStr("%2.15f" ,learning_rate)
        fname = "%scdlc-mlp-simple-loss-values-%s-%s-%s-%s-%d.bin" % (path,optimName,lrStr,momentumStr,initStr,maxEpoch)
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
luapath = './lua-model/'
compositionalMethod = 'additive'
lrStrMdl = mres.floatToStr("%f",1e-3)
mmtStrMdl = mres.floatToStr("%f",0.2)
epochMdl = 500
batchSizeMdl = 100
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'


#luaModelEngFile = "./lua-model/english.1000.tok.model.t7"
#luaModelTrFile = "./lua-model/turkish.1000.tok.model.t7"
luaEmbedPriFile = luapath + "english.1000.tok.embed.t7"
luaEmbedSecFile = luapath + "turkish.1000.tok.embed.t7"
#luaVocabEngFile = "./lua-model/english.1000.tok.en-tu.vocab"
#luaVocabTrFile = "./lua-model/turkish.1000.tok.en-tu.vocab"
luaVocabPriFile = luapath + "eng-1000.tok.vocab.ascii.txt"
luaVocabSecFile = luapath + "tr-1000.tok.vocab.ascii.txt"

from torch.utils.serialization import load_lua
luaEmbedPri = load_lua(luaEmbedPriFile,unknown_classes=True)
luaEmbedSec = load_lua(luaEmbedSecFile,unknown_classes=True)
#luaVocabEngPri = load_lua(luaVocabEngFile,unknown_classes=True)
#luaVocabTrPri = load_lua(luaVocabTrFile,unknown_classes=True)
#luaModelEngPri = load_lua(luaModelEngFile,unknown_classes=True)
#luaModelTrPri = load_lua(luaModelTrFile,unknown_classes=True)
luaEmbedVecPri =load_lua(luapath + "eng-art-vectors.bin",unknown_classes=True)
luaEmbedVecSec =load_lua(luapath + "tr-art-vectors.bin",unknown_classes=True)

luaVocabPri = {}
with open(luaVocabPriFile, 'r') as f:
    ln = f.readline()
    ln = f.readline()
    ln = f.readline()
    ln = f.readline()
    ln = f.readline()
    while True:
        key = f.readline()
        if not key:
            break
        ln = f.readline()
        value = f.readline()
        ln = f.readline()
        ln = f.readline()
        luaVocabPri[key.split()[0]] = int(value.split()[0])

    luaVocabSec = {}
    with open(luaVocabSecFile, 'r') as f:
        ln = f.readline()
        ln = f.readline()
        ln = f.readline()
        ln = f.readline()
        ln = f.readline()
        while True:
            key = f.readline()
            if not key:
                break
            ln = f.readline()
            value = f.readline()
            ln = f.readline()
            ln = f.readline()
            luaVocabSec[key.split()[0]] = int(value.split()[0])


#luaModelEngPri = torch.load( open( luaModelEngFile, "rb" ) )
#luaEmbedEngPri = torch.load( open( luaEmbedEngFile, "rb" ) )
#luaVocabEngPri = torch.load( open( luaVocabEngFile, "rb" ) )
#luaModelTrPri = torch.load( open( luaModelTrFile, "rb" ) )
#luaEmbedTrPri = torch.load( open( luaEmbedTrFile, "rb" ) )
#luaVocabTrPri = torch.load( open( luaVocabTrFile, "rb" ) )


embedding_dim = 64
batchSize = 1
max_epoch = 50 # 300
learning_rate_decay = 0.01
threshold = 100
folds = 1 # 10
test_per = 0.25
train_per = 1
#init_weight = 1e-1 #0.001

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
results = {}
results['train'] = []
results['test'] = []

fResOut = open(path + compositionalMethod + '-' + 'cdlc-mlp-simple-run.txt', "w")
ftestRes = open("cdlc-mlp-simple-run.txt", "w")
ftestRes.write("classname,fold,lr,momentum,max_epoch,all_positives,all_negatives,correct,predicted_positives,predicted_negatives,"
               "true_positives,true_negatives,precision,recall,f1_score,score,accuracy,error_rate\n")

load = True
lua = False
otherClass = False

for classname in classes:
    if load == False:
        print(datetime.datetime.today())
        if lua == True:
            print("Lua model will be used")
            positivePri, negativePri = getData(data_pathPri, classname, luaVocabPri)
            positiveSec, negativeSec = getData(data_pathSec, classname, luaVocabSec)
            modelLoaded.embeddings_pri.weight.data.copy_(luaEmbedPri)
            modelLoaded.embeddings_sec.weight.data.copy_(luaEmbedSec)
        else:
            positivePri, negativePri = getData(data_pathPri,classname,vocabPri)
            positiveSec, negativeSec = getData(data_pathSec, classname, vocabSec)

        print("%s loaded %s" % (classname,data_pathPri) )
        all_rawPri = torch.cat([positivePri,negativePri], 0)
        targetsPri = torch.cat([torch.Tensor(positivePri.size()[0]).fill_(1),torch.Tensor(negativePri.size()[0]).fill_(0)],0 )
        allPri = autograd.Variable(modelLoaded.forwardPri(all_rawPri.long().cuda(),"add"))
        allPriDocVecs = autograd.Variable(modelLoaded.forwardPri(all_rawPri.long().cuda(),"minmax"))
        print(datetime.datetime.today())
        all_positivesPri=positivePri.size()[0]  # 122
        all_negativesPri=negativePri.size()[0]  # 118

        print("%s loaded %s" % (classname,data_pathSec) )
        positiveSec = positiveSec[0:math.floor(positiveSec.size()[0]*test_per),]
        negativeSec = negativeSec[0:math.floor(negativeSec.size()[0]*test_per),]
        all_rawSec = torch.cat([positiveSec,negativeSec], 0)
        targetsSec = torch.cat([torch.Tensor(positiveSec.size()[0]).fill_(1),torch.Tensor(negativeSec.size()[0]).fill_(0)],0 )
        allSec = autograd.Variable(modelLoaded.forwardSec(all_rawSec.long().cuda(),"add"))
        allSecDocVecs = autograd.Variable(modelLoaded.forwardSec(all_rawSec.long().cuda(),"minmax"))
        all_positivesSec=positiveSec.size()[0]  # 27
        all_negativesSec=negativeSec.size()[0]  # 26

        if lua == True:
            pickle.dump(allPri, open("allPri-mlp-simple-lua.pck", 'wb'))
            pickle.dump(allSec, open("allSec-mlp-simple-lua.pck", 'wb'))
            pickle.dump(targetsPri, open("targetsPri-mlp-simple-lua.pck", 'wb'))
            pickle.dump(targetsSec, open("targetsSec-mlp-simple-lua.pck", 'wb'))
        else:
            pickle.dump(allPri, open("allPri-mlp-simple.pck", 'wb'))
            pickle.dump(allSec, open("allSec-mlp-simple.pck", 'wb'))
            pickle.dump(allPriDocVecs, open("allPriDocVecs-mlp-simple.pck", 'wb'))
            pickle.dump(allSecDocVecs, open("allSecDocVecs-mlp-simple.pck", 'wb'))
            pickle.dump(targetsPri, open("targetsPri-mlp-simple.pck", 'wb'))
            pickle.dump(targetsSec, open("targetsSec-mlp-simple.pck", 'wb'))

    else:
        if lua == True:
            allPri = pickle.load(open("allPri-mlp-simple-lua.pck", 'rb'))
            allSec = pickle.load(open("allSec-mlp-simple-lua.pck", 'rb'))
            targetsPri = pickle.load(open("targetsPri-mlp-simple-lua.pck", 'rb'))
            targetsSec = pickle.load(open("targetsSec-mlp-simple-lua.pck", 'rb'))
            allPri = autograd.Variable(luaEmbedVecPri.float().cuda())
            allSec = autograd.Variable(luaEmbedVecSec.float().cuda())
            print("%s Pickle Lua files loaded " % (classname))
        else:
            allPri = pickle.load(open("allPri-mlp-simple.pck", 'rb'))
            allSec = pickle.load(open("allSec-mlp-simple.pck", 'rb'))
            allPriDocVecs = pickle.load(open("allPriDocVecs-mlp-simple.pck", 'rb'))
            allSecDocVecs = pickle.load(open("allSecDocVecs-mlp-simple.pck", 'rb'))
            targetsPri = pickle.load(open("targetsPri-mlp-simple.pck", 'rb'))
            targetsSec = pickle.load(open("targetsSec-mlp-simple.pck", 'rb'))
            print("%s Pickle files loaded " % (classname) )
            allPri = autograd.Variable(scaleTensor(allPri.data).cuda())
            allSec = autograd.Variable(scaleTensor(allSec.data).cuda())

        all_positivesPri= 122
        all_negativesPri= 118
        all_positivesSec= 27
        all_negativesSec= 26

    if otherClass == True:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn import svm
        from sklearn.datasets import make_blobs

        X = allPri.data.cpu().numpy()
        y = targetsPri.numpy()
        for i in range(5):
            clfRF = RandomForestClassifier(n_estimators=100, max_depth=None,max_features=None, min_samples_split = 2, random_state = 0)
            scoresRF = cross_val_score(clfRF, X, y,cv=(i+1)*10)
            clfAB = AdaBoostClassifier(n_estimators=100)
            scoresAB = cross_val_score(clfAB, X, y, cv=(i + 1) * 10)
            clfGB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 0)
            scoresGB = cross_val_score(clfGB, X, y, cv=(i + 1) * 10)
            clfSVM = svm.SVC()
            scoresSVM = cross_val_score(clfSVM, X, y, cv=(i + 1) * 10)
            print("cv %d RF mean %f - AB mean %f - GB mean %f - SVM mean %f" % ((i+1)*10,scoresRF.mean(),scoresAB.mean(),scoresGB.mean(),scoresSVM.mean()))
        exit()


    optimizerList = ["SGD","RMSprop","Adadelta","Adagrad","Adam","Adamax","ASGD"]
    optimizerList = ["SGD","RMSprop","Adam"]
    momentumList=[0,0.5,1]
    lrList= [1,1e-3,1e-5,1e-7]
    initWeightList = [1e-1,1e-3,1e-5]
    embedding_dim = 64
    hidden_dim = 64

    prmList = []
    prmList.append(["RMSprop", 0.001, 0.5, 0.000001, 5000])

    results['train'].append("class : " + classname)
    results['test'].append("class : " + classname)

    for prm in prmList:
        optim = prm[0]
        lr = prm[1]
        momentum = prm[2]
        initWeight = prm[3]
        max_epoch = prm[4]
        results['train'].append("optimizer : " + optim)
        results['test'].append("optimizer : " + optim)
        classifierTrained1 = trainClassifierNew(allPriDocVecs, targetsPri, allSecDocVecs, targetsSec, lr, momentum, max_epoch, initWeight,True, optim, results)
        #classifierTrained2 = trainClassifierNew(allPri, targetsPri, allSec, targetsSec, lr, momentum,max_epoch, initWeight, True, optim, results)
#    for optim in optimizerList:
#        results['train'].append("optimizer : " + optim)
#        results['test'].append("optimizer : " + optim)
#        for momentum in momentumList:
#            for lr in lrList:
#                for initWeight in initWeightList:
#                    classifierTrained=trainClassifier(allPri,targetsPri,allSec,targetsSec,lr,momentum,max_epoch,initWeight,True,optim,results)

fResOut.write("train\n")
for res in results['train']:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

fResOut.write("test\n")
for res in results['test']:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")


ftestRes.close()
fResOut.close()
print("End of classifier test")

