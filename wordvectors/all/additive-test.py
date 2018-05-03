
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
        self.longest_sent = 0 #self.longestSentLength(self.file)
        self.data = torch.LongTensor(1) #self.shape_data()
        

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
    
class BiLingual(nn.Module):
    
    def __init__(self, vocab_size_pri,vocab_size_sec ,embedding_dim):
        super(BiLingual, self).__init__()
        self.embeddings_pri = nn.Embedding(vocab_size_pri, embedding_dim)
        self.embeddings_sec = nn.Embedding(vocab_size_sec, embedding_dim)

    def init_weights(self):
        initrange = 0.01
        init.uniform(self.embeddings_pri.weight,-1*initrange,initrange)
        init.uniform(self.embeddings_sec.weight,-1*initrange,initrange)

    def cAdd(self,embeds):
        btch_len = embeds.size()[0]
        sntc_len = embeds.size()[1]
        ret = []
        for i in range(btch_len):
            #splt=torch.split(embeds[i],sntc_len,1)
            #tot2 = autograd.Variable(torch.zeros(embedding_dim).cuda(),requires_grad=True)
            #for j in range(sntc_len):
            #    tot2 = tot2 + embeds[i][j]
            tot = torch.sum(embeds[i],0)[0]
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

def test_model(model, inputs_pri, inputs_sec):
    inputPri = autograd.Variable(inputs_pri.cuda())
    inputSec = autograd.Variable(inputs_sec.cuda())
    outputPri = model.forwardPri(inputPri)
    outputSec = model.forwardSec(inputSec)
    all_rootsPri = outputPri.data.float()
    all_rootsSec = outputSec.data.float()

    final = True
    list1 = {}
    list2 = {}
    score = 0
    #f2 = open(compositionalMethod+"-all-sent-vect-closest.txt", "w")
    for idxPri in range(all_rootsPri.size()[0]):
        closest = idxPri
        dist1 = torch.dist(all_rootsPri[idxPri],all_rootsSec[closest])
        for idxSec in range(idxPri,all_rootsSec.size()[0]):
            dist2 = torch.dist(all_rootsPri[idxPri], all_rootsSec[idxSec])
            diff = dist1 - dist2
            if dist2 < dist1:
                closest = idxSec
                dist1 = dist2

        if idxPri == closest:
            score = score + 1
            list2[idxPri] = closest
        else:
            if final == True:
                list1[idxPri] = closest
        #f2.write("-------\n")
        #f2.write("%d - %s\n" % (idxPri,corpus_pri.sentences[idxPri]))
        #f2.write("%d - %s\n" % (closest,corpus_sec.sentences[closest]))

    #f2.close()
    print("Test Score: %d / %d " % (score,all_rootsPri.size()[0]))
    return list1, list2,score,all_rootsPri.size()[0]

def adjust_learning_rate(optimizer, epoch,threshold,lr_init,lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (lr_decay_rate ** (epoch // threshold))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def loadAndTestModel(fout,fullModelNamePth,corpus_pri,corpus_sec):
    model = BiLingual(vocab_size_pri+1,vocab_size_sec+1,embedding_dim).cuda()
    model.init_weights()
    model.load_state_dict(torch.load(fullModelNamePth))
    print("model file : %s loaded" % fullModelNamePth)
    number_of_sentences = math.floor((len(corpus_pri.sentences)/batch_size)*batch_size)

    inds = torch.range(1, number_of_sentences, 1).long()
    inputs_pri = torch.zeros(test_size, corpus_pri.data.size()[1]).long()
    inputs_sec = torch.zeros(test_size, corpus_sec.data.size()[1]).long()

    totTrial = 0
    totScore = 0

    for trial in range(0,10):
        shuffle = torch.randperm(inds.size()[0])
        for i in range(test_size):
            inputs_pri[i] = corpus_pri.data[shuffle[i]]
            inputs_sec[i] = corpus_sec.data[shuffle[i]]
        dummy1,dummy2,testScore,total = test_model(model,inputs_pri,inputs_sec)
        print("%s,%d,%s,%s,%d,%d,%d\n" % (compositionalMethod, trial+1, lrStr, mmtStr, max_epoch, testScore, total))
        fout.write("%s,%d,%s,%s,%d,%d,%d\n" % (compositionalMethod,trial+1,lrStr,mmtStr,max_epoch,testScore,total))
        fout.flush()
        totTrial += 1
        totScore += testScore
    print("trial : %d  avg : %f" % (totTrial,totScore/totTrial))

def loadCorpusData(corpusFileName):
    print('Corpus file %s' % corpusFileName)
    priCorpusFile = corpusFileName + ".corpus"
    return pickle.load(open(priCorpusFile, "rb"))


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
compositionalMethod = 'additive'
corpusPri = Corpus(path,fnamePri)
corpusSec = Corpus(path,fnameSec)
#corpusPri.data.cuda()
#corpusSec.data.cuda()

corpusPri = loadCorpusData(corpusPri.file)
corpusSec = loadCorpusData(corpusSec.file)

vocab_size_pri = corpusPri.vocab_idx
vocab_size_sec = corpusSec.vocab_idx
embedding_dim = 64
batch_size = 10000 # 100
max_epoch = 1000 # 500
#learning_rate = 0.001
learning_rate_decay = 0.1
threshold = 50
last_epoch = 1000
test_size = 1000
test_run = 10

momentlist = [0.2]  # ok
lrlist = [0.001] # ok

print("Primary name : %s  # of sent. : %d ,# of vocabs :%d, longest sent : %d " % (fnamePri,len(corpusPri.sentences),vocab_size_pri,corpusPri.longest_sent))
print("Secondary name : %s  # of sent. : %d ,# of vocabs :%d, longest sent : %d " % (fnameSec,len(corpusSec.sentences),vocab_size_sec,corpusSec.longest_sent))

ftestRes = open(compositionalMethod + "-all-test-run.txt", "w")
ftestRes.write("name,epoch,lr,momentum,max_epoch,test score,total\n")

for mmt in momentlist:
    for lr in lrlist:
        lrStr = ("%2.15f" % lr).rstrip('0')
        mmtStr = ("%2.5f" % mmt).rstrip('0')
        prmStr = "%s-%s-%d-%d-%d" % (lrStr,mmtStr,max_epoch,batch_size,last_epoch)
        fullModelNamePth = "%s%s-%s-%s.pth" % (path, compositionalMethod, modelName, prmStr)
        modelTrained = loadAndTestModel(ftestRes,fullModelNamePth,corpusPri,corpusSec)

ftestRes.close()
print("End Program")
