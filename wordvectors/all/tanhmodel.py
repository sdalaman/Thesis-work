
# coding: utf-8


import os
import sys
import math
#import matplotlib
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import datetime

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

class LossValues(object):
    def __init__(self):
        self.x = []
        self.y1 = []
        self.y2 = []
        self.mean = []

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path,fname):
        self.sentences = []
        self.vocab_idx = 0
        self.vocab_map = {'<pad>': 0}
        self.id2vocab_map = {}
        self.dictionary = Dictionary()
        self.file = path + fname
        self.longest_sent = self.longestSentLength(self.file)
        self.data = self.shape_data()
        

    def longestSentLength(self,file):
        assert os.path.exists(file)
        # Add words to the dictionary
        max_len = 0
        with open(file, 'r') as f:
            for line in f:
                self.sentences.append(line) 
                words = line.split()
                if max_len < len(words):
                    max_len = len(words)
        return max_len

    def padding(self,sentence):
        new_sentence = []
        for i in range(0 , self.longest_sent):
            new_sentence.append('<pad>')
        j = 1
        for i in range((self.longest_sent - len(sentence) + 1) , self.longest_sent+1):
            new_sentence[i-1] = sentence[j-1]
            j = j + 1
        return new_sentence

    def reverseMap(self):
        for key in self.vocab_map:
            self.id2vocab_map[self.vocab_map[key]] = key

    def shape_data(self):
        x = torch.zeros(len(self.sentences),self.longest_sent)
        for i in range(0,len(self.sentences)):
            words = self.sentences[i].split()
            words = self.padding(words)
            for j in range(0,len(words)):
                if self.vocab_map.get(words[j]) == None:
                    self.vocab_idx = self.vocab_idx + 1
                    self.vocab_map[words[j]] = self.vocab_idx
                x[i][j] = self.vocab_map[words[j]]
        print("Number of words = %d" % self.vocab_idx)
        self.no_of_words = self.vocab_idx
        self.reverseMap()
        return x.long()
        
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class args(object):
    pass
    

#def xavier_init(t):
    #This seems to be the recommended distribution for weight initialization"
#    n = max(t.size())
#    return t.normal_(std=n ** -.5)

class BiLingual(nn.Module):
    
    def __init__(self, vocab_size_pri,vocab_size_sec ,embedding_dim):
        super(BiLingual, self).__init__()
        self.embeddings_pri = nn.Embedding(vocab_size_pri, embedding_dim)
        self.embeddings_sec = nn.Embedding(vocab_size_sec, embedding_dim)
        self.tanh = nn.Tanh()

    def init_weights(self):
        initrange = 0.01
        init.uniform(self.embeddings_pri.weight,-1*initrange,initrange)
        init.uniform(self.embeddings_sec.weight,-1*initrange,initrange)

    def cAdd(self,embeds):
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            tot = autograd.Variable(torch.zeros(embedding_dim).cuda(),requires_grad=True)
            for j in range(sntc_len - 1):
                tot = tot + self.tanh(embeds[i][j] + embeds[i][j+1])
            #tot = torch.sum(embeds[i],0)[0]
            aa = embeds[i][:, 0:(embeds[i].size()[1] - 1)] + embeds[i][:, 1:embeds[i].size()[1]]
            bb = torch.tanh(aa)
            tot2 = torch.sum(bb, 1)
            ret.append(tot)
        ret=torch.stack(ret,0)
        return ret

    def forwardPriOld(self, inputs):
        out_pri1 = self.embeddings_pri(inputs)
        return out_pri1[0][1]
        out = autograd.Variable(torch.zeros(out_pri1.data.size()[0],out_pri1.data.size()[2]).cuda())
        for u in range(out_pri1.data.size()[0]):
            for t in range(out_pri1.data.size()[1]):
                out[u] = out[u] + out_pri1[u][t]
        return out

    def forwardSecOld(self, inputs):
        out_pri1 = self.embeddings_sec(inputs)
        return out_pri1[0][1]
        out = autograd.Variable(torch.zeros(out_pri1.data.size()[0],out_pri1.data.size()[2]).cuda())
        for u in range(out_pri1.data.size()[0]):
            for t in range(out_pri1.data.size()[1]):
                out[u] = out[u] + out_pri1[u][t]
        return out

    def forwardPri(self, inputs):
        embeds_pri = self.embeddings_pri(inputs)
        out_pri = self.cAdd(embeds_pri)
        return out_pri
    
    def forwardSec(self, inputs):
        embeds_sec = self.embeddings_sec(inputs)
        out_sec = self.cAdd(embeds_sec)
        return out_sec

def test_model(model, corpus_pri, corpus_sec):
    inputs_pri = corpus_pri.data
    inputs_sec = corpus_sec.data
    inputPri = autograd.Variable(inputs_pri[0:1000].cuda())
    inputSec = autograd.Variable(inputs_sec[0:1000].cuda())
    outputPri = model.forwardPri(inputPri)
    outputSec = model.forwardSec(inputSec)
    all_rootsPri = outputPri.data.float()
    all_rootsSec = outputSec.data.float()

    final = True
    list1 = {}
    list2 = {}
    score = 0
    #f1 = open(compositionalMethod+"-1000-sent-vect-dist.txt","w")
    f2 = open(compositionalMethod+"-all-sent-vect-closest.txt", "w")
    for idxPri in range(all_rootsPri.size()[0]):
        closest = idxPri
        dist1 = torch.dist(all_rootsPri[idxPri],all_rootsSec[closest])
        for idxSec in range(all_rootsSec.size()[0]):
            dist2 = torch.dist(all_rootsPri[idxPri], all_rootsSec[idxSec])
            diff = dist1 - dist2
            #f1.write("%d ,%d - %s,%f,%f,%f\n" % (idxPri,corpus_pri.id2vocab_map[idxPri],idxSec,corpus_sec.id2vocab_map[idxSec], dist1, dist2, diff))
            if dist2 < dist1:
                closest = idxSec
                dist1 = dist2

        if idxPri == closest:
            score = score + 1
            list2[idxPri] = closest
        else:
            if final == True:
                list1[idxPri] = closest
        f2.write("-------\n")
        f2.write("%d - %s\n" % (idxPri,corpus_pri.sentences[idxPri]))
        f2.write("%d - %s\n" % (closest,corpus_sec.sentences[closest]))

    #f1.close()
    f2.close()
    print("Test Score: %d / %d " % (score,all_rootsPri.size()[0]))
    return list1, list2,score,all_rootsPri.size()[0]

def adjust_learning_rate(optimizer, epoch,threshold,lr_init,lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (lr_decay_rate ** (epoch // threshold))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_model(fout,learning_rate,n_momentum):
    lossVal = LossValues()
    loss_function = nn.L1Loss()
    model = BiLingual(vocab_size_pri+1,vocab_size_sec+1,embedding_dim).cuda()
    model.init_weights()
    pri_parameters = [
        {'params': model.embeddings_pri.parameters()}
    ]
    sec_parameters = [
        {'params': model.embeddings_sec.parameters()}
    ]

    lr_print = learning_rate
    lrStr  = ("%2.15f" % learning_rate).rstrip('0')
    mmtStr = ("%2.5f" % n_momentum).rstrip('0')


    optimizerPri = optim.RMSprop(pri_parameters, lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=n_momentum, centered=False)
    optimizerSec = optim.RMSprop(sec_parameters, lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=n_momentum, centered=False)

    number_of_sentences = math.floor((len(corpusPri.sentences)/batch_size)*batch_size)
    losses = []
    meanlosses = []
    ppPri = []
    ppSec = []

    totalCnt = 0
    totalLoss = 0
    allbegin =  datetime.datetime.today()
    for epoch in range(max_epoch):
        inds = torch.range(1, number_of_sentences,batch_size).long()
        shuffle = torch.randperm(inds.size()[0])
        epochbegin = datetime.datetime.today()
        print('epoch %d - lr %s momentum %s' % (epoch+1, lrStr, mmtStr))
        for j in range(int(number_of_sentences/batch_size)):

            batchbegin = datetime.datetime.today()
            start = inds[shuffle[j]]-1
            endd = inds[shuffle[j]]+batch_size-1
            #print('epoch %d step %d - lr %s momentum %s' % (epoch+1, j,lrStr,mmtStr))
            #print('start %d end  %d' % (start,endd))
            inputPri = autograd.Variable(corpusPri.data[start:endd]).cuda()
            inputSec = autograd.Variable(corpusSec.data[start:endd]).cuda()

            outputPri1 = model.forwardPri(inputPri)
            outputSec1 = model.forwardSec(inputSec)
            outputSec1T = autograd.Variable(outputSec1.data.float(),requires_grad=False)
            model.zero_grad()
            lossPri = loss_function(outputPri1,outputSec1T)
            lossPri.backward()
            optimizerPri.step()

            outputPri2 = model.forwardPri(inputPri)
            outputSec2 = model.forwardSec(inputSec)
            outputPri2T= autograd.Variable(outputPri2.data.float(),requires_grad=False)
            model.zero_grad()
            lossSec = loss_function(outputSec2,outputPri2T)
            lossSec.backward()
            optimizerSec.step()

            losses.append(lossPri.data[0])
            losses.append(lossSec.data[0])
            lossVal.y1.append(lossPri.data[0])
            lossVal.y2.append(lossSec.data[0])
            totalCnt += 1
            totalLoss += lossPri.data[0] + lossSec.data[0]
            mean = totalLoss / totalCnt
            #mean = torch.mean(torch.Tensor(losses))
            meanlosses.append(mean)
            print("epoch %d-%d pri loss %f - sec loss %f - mean %f " % (epoch+1,j+1,lossPri.data[0],lossSec.data[0],mean))
            batchend = datetime.datetime.today()

        if (epoch+1) % threshold == 0 and epoch != 0:
            dummy1,dummy2,testScore,total = test_model(model,corpusPri,corpusSec)
            fout.write("%s,%d,%s,%s,%d,%d,%d\n" % (compositionalMethod,epoch+1,lrStr,mmtStr,max_epoch,testScore,total))
            fout.flush()
            lossVal.x = range(max_epoch * int(number_of_sentences / batch_size))
            lossVal.mean = meanlosses
            prmStr = "%s-%s-%d-%d-%d" % (lrStr, mmtStr,max_epoch, batch_size,epoch+1)
            fname = "%s%s-all-loss-values-%s.bin" % (path, compositionalMethod, prmStr)
            fh = open(fname, 'wb')  # Save model file as pickle
            pickle.dump(lossVal, fh)
            fh.close()
            saveModel(model,str(epoch+1))
            print("Model Saved")

        epochend = datetime.datetime.today()
        print(" epoch duration = %s" % (epochend - epochbegin))

    allend = datetime.datetime.today()
    print(" all duration = %s" % (allend - allbegin))

    lossVal.x = range(max_epoch*int(number_of_sentences/batch_size))
    lossVal.mean = meanlosses
    prmStr = "%s-%s-%d-%d-%d" % (lrStr, mmtStr, max_epoch,batch_size,max_epoch)
    fname = "%s%s-all-loss-values-%s.bin" % (path,compositionalMethod,prmStr)
    fh = open(fname, 'wb')  # Save model file as pickle
    pickle.dump(lossVal, fh)
    fh.close()
    return model


def saveModel(model,epc):
    fullModelNamePth = "%s%s-%s-%s-%s.pth" % (path,compositionalMethod,modelName,prmStr,epc)
    print('Model file %s' % fullModelNamePth)
    torch.save(model.state_dict(), fullModelNamePth)
    #fh = open(fullModelNamePck, 'wb')  # Save model file as pickle
    #pickle.dump(model, fh)
    #fh.close()

def saveModelPck(model,epc):
    fullModelNamePck = "%s%s-%s-%s-%s.pck" % (path,compositionalMethod,modelName,prmStr,epc)
    print('Model file (Pck) %s' % fullModelNamePck)
    fh = open(fullModelNamePck, 'wb')  # Save model file as pickle
    pickle.dump(model, fh)
    fh.close()

def saveCorpusData():
    print('Corpus file %s' % corpusPri.file)
    priCorpusFile = corpusPri.file + ".corpus"
    fh = open(priCorpusFile, 'wb')
    pickle.dump(corpusPri, fh)  # Save primary vocab file as pickle
    fh.close()
    print('Corpus file %s' % corpusSec.file)
    secCorpusFile = corpusSec.file + ".corpus"
    fh = open(secCorpusFile, 'wb')
    pickle.dump(corpusSec, fh)  # Save primary vocab file as pickle
    fh.close()
    fullVocabFilePri = corpusPri.file + ".vocab"
    fh = open(fullVocabFilePri, 'wb')
    pickle.dump(corpusPri.vocab_map, fh)  # Save primary vocab file as pickle
    fh.close()
    fullVocabFileSec = corpusSec.file + ".vocab"
    fh = open(fullVocabFileSec, 'wb')
    pickle.dump(corpusSec.vocab_map, fh)  # Save secondary vocab file as pickle
    fh.close()


# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

###############################################################################
# Load data
###############################################################################
fnamePri = 'english.all.tok'
fnameSec = 'turkish.all.tok'
modelName = 'modelAllEnTr'
path = './'
compositionalMethod = 'tanh'
corpusPri = Corpus(path,fnamePri)
corpusSec = Corpus(path,fnameSec)
corpusPri.data.cuda()
corpusSec.data.cuda()

saveCorpusData()

vocab_size_pri = corpusPri.vocab_idx
vocab_size_sec = corpusSec.vocab_idx
embedding_dim = 64
batch_size = 100 # 100
max_epoch = 50 # 500
#learning_rate = 0.001
learning_rate_decay = 0.1
threshold = 10

momentlist = [0.2]
lrlist = [0.001]

print("Primary name : %s  # of sent. : %d ,# of vocabs :%d, longest sent : %d " % (fnamePri,len(corpusPri.sentences),vocab_size_pri,corpusPri.longest_sent))
print("Secondary name : %s  # of sent. : %d ,# of vocabs :%d, longest sent : %d " % (fnameSec,len(corpusSec.sentences),vocab_size_sec,corpusSec.longest_sent))


ftestRes = open(compositionalMethod + "-all-run.txt", "w")
ftestRes.write("name,epoch,lr,momentum,max_epoch,test score,total\n")

for mmt in momentlist:
    for lr in lrlist:
        lrStr = ("%2.15f" % lr).rstrip('0')
        mmtStr = ("%2.5f" % mmt).rstrip('0')
        prmStr = "%s-%s-%d-%d" % (lrStr,mmtStr,max_epoch,batch_size)
        #fullModelNamePth = path + compositionalMethod + '-' + 'modelAllEnTr-'+prmStr+".pth"
        #fullModelNamePck = path + compositionalMethod + '-' + 'modelAllEnTr-'+prmStr+".pck"
        #fullVocabFilePri = path + compositionalMethod + '-' + 'english.AllEnTr-'+prmStr+".vocab"
        #fullVocabFileSec = path + compositionalMethod + '-' + 'turkish.AllEnTr-'+prmStr+".vocab"
        modelTrained = train_model(ftestRes,lr,mmt)
        dummy1,dummy2,testScore,total= test_model(modelTrained,corpusPri,corpusSec)
        ftestRes.write("%s,%d,%s,%s,%d,%d,%d\n"% (compositionalMethod,max_epoch,lrStr,mmtStr,max_epoch,testScore,total))
        ftestRes.flush()
        print("End Training")
        saveModel(modelTrained,"end")
        saveModelPck(modelTrained,"end")
        print("Model Saved")

ftestRes.close()
print("End Program")