
import os
import sys
import math
# import matplotlib
import numpy as np
# import matplotlib.pyplot as plt
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
    def __init__(self, path ,fname):
        self.sentences = []
        self.vocab_idx = 0
        self.vocab_map = {'<pad>': 0}
        self.dictionary = Dictionary()
        self.file = path + fname
        self.longest_sent = self.longestSentLength(self.file)
        self.data = self.shape_data()


    def longestSentLength(self ,file):
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

    def padding(self ,sentence):
        new_sentence = []
        for i in range(0 , self.longest_sent):
            new_sentence.append('<pad>')
        j = 1
        for i in range((self.longest_sent - len(sentence) + 1) , self. longest_sent +1):
            new_sentence[ i -1] = sentence[ j -1]
            j = j + 1
        return new_sentence

    def shape_data(self):
        x = torch.zeros(len(self.sentences) ,self.longest_sent)
        for i in range(0 ,len(self.sentences)):
            words = self.sentences[i].split()
            words = self.padding(words)
            for j in range(0 ,len(words)):
                if self.vocab_map.get(words[j]) == None:
                    self.vocab_idx = self.vocab_idx + 1
                    self.vocab_map[words[j]] = self.vocab_idx
                x[i][j] = self.vocab_map[words[j]]
        print("Number of words = %d" % self.vocab_idx)
        self.no_of_words = self.vocab_idx
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

    def __init__(self, vocab_size_pri ,vocab_size_sec ,embedding_dim):
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
            #splt = torch.split(embeds[i], sntc_len, 1)
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


def test_model(model, inputs_pri, inputs_sec):
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
    f1 = open(compositionalMethod + "-sent-vect-dist.txt", "w")
    f2 = open(compositionalMethod + "-sent-vect-closest.txt", "w")
    for idxPri in range(all_rootsPri.size()[0]):
        closest = idxPri
        dist1 = torch.dist(all_rootsPri[idxPri], all_rootsSec[closest])
        for idxSec in range(all_rootsSec.size()[0]):
            dist2 = torch.dist(all_rootsPri[idxPri], all_rootsSec[idxSec])
            diff = dist1 - dist2
            f1.write("%d,%d,%f,%f,%f\n" % (idxPri, idxSec, dist1, dist2, diff))
            if dist2 < dist1:
                closest = idxSec
                dist1 = dist2

        if idxPri == closest:
            score = score + 1
            list2[idxPri] = closest
        else:
            if final == True:
                list1[idxPri] = closest
        f2.write("%d,%d\n" % (idxPri, closest))

    f1.close()
    f2.close()
    print("Test Score: %d / %d " % (score, all_rootsPri.size()[0]))
    return list1, list2, score, all_rootsPri.size()[0]


def adjust_learning_rate(optimizer, epoch, threshold, lr_init, lr_decay_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (lr_decay_rate ** (epoch // threshold))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_model(fout, learning_rate, n_momentum):
    lossVal = LossValues()
    loss_function = nn.L1Loss()
    model = BiLingual(vocab_size_pri + 1, vocab_size_sec + 1, embedding_dim).cuda()
    model.init_weights()
    pri_parameters = [
        {'params': model.embeddings_pri.parameters()}
    ]
    sec_parameters = [
        {'params': model.embeddings_sec.parameters()}
    ]

    lr_print = learning_rate
    lrStr = ("%2.15f" % learning_rate).rstrip('0')
    mmtStr = ("%2.5f" % n_momentum).rstrip('0')

    optimizerPri = optim.RMSprop(pri_parameters, lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,
                                 momentum=n_momentum, centered=False)
    optimizerSec = optim.RMSprop(sec_parameters, lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,
                                 momentum=n_momentum, centered=False)

    number_of_sentences = math.floor((len(corpusEng.sentences) / batch_size) * batch_size)
    losses = []
    meanlosses = []
    ppPri = []
    ppSec = []

    allbegin = datetime.datetime.today()
    for epoch in range(max_epoch):
        inds = torch.range(1, number_of_sentences, batch_size).long()
        shuffle = torch.randperm(inds.size()[0])
        epochbegin = datetime.datetime.today()
        for j in range(int(number_of_sentences / batch_size)):
            batchbegin = datetime.datetime.today()
            start = inds[shuffle[j]] - 1
            endd = inds[shuffle[j]] + batch_size - 1
            print('epoch %d step %d - lr %s momentum %s' % (epoch, j, lrStr, mmtStr))
            # print('start %d end  %d' % (start,endd))
            inputEng = autograd.Variable(corpusEng.data[start:endd]).cuda()
            inputTr = autograd.Variable(corpusTr.data[start:endd]).cuda()

            outputPri1 = model.forwardPri(inputEng)
            outputSec1 = model.forwardSec(inputTr)
            outputSec1T = autograd.Variable(outputSec1.data.float(), requires_grad=False)
            model.zero_grad()
            lossPri = loss_function(outputPri1, outputSec1T)
            lossPri.backward()
            optimizerPri.step()

            outputPri2 = model.forwardPri(inputEng)
            outputSec2 = model.forwardSec(inputTr)
            outputPri2T = autograd.Variable(outputPri2.data.float(), requires_grad=False)
            model.zero_grad()
            lossSec = loss_function(outputSec2, outputPri2T)
            lossSec.backward()
            optimizerSec.step()

            losses.append(lossPri.data[0])
            losses.append(lossSec.data[0])
            lossVal.y1.append(lossPri.data[0])
            lossVal.y2.append(lossSec.data[0])
            mean = torch.mean(torch.Tensor(losses))
            meanlosses.append(mean)
            print("pri loss %f - sec loss %f - mean %f " % (lossPri.data[0], lossSec.data[0], mean))
            batchend = datetime.datetime.today()

        if epoch % threshold == 0 and epoch != 0:
            dummy1, dummy2, testScore, total = test_model(model, corpusEng.data, corpusTr.data)
            fout.write(
                "%s,%d,%s,%s,%d,%d,%d\n" % (compositionalMethod, epoch, lrStr, mmtStr, max_epoch, testScore, total))

        epochend = datetime.datetime.today()
        print(" epoch duration = %s" % (epochend - epochbegin))

    allend = datetime.datetime.today()
    print(" all duration = %s" % (allend - allbegin))

    lossVal.x = range(max_epoch * int(number_of_sentences / batch_size))
    lossVal.mean = meanlosses
    prmStr = "%s-%s-%d-%d" % (lrStr, mmtStr, max_epoch, batch_size)
    fname = "%s%s-loss-values-%s.bin" % (path, compositionalMethod, prmStr)
    fh = open(fname, 'wb')  # Save model file as pickle
    pickle.dump(lossVal, fh)
    fh.close()
    return model


def saveModel(model):
    print('Primary file %s' % corpusEng.file)
    print('Secondary file %s' % corpusTr.file)
    print('Model file %s' % fullModelNamePth)
    print('Primary Vocab Model file %s' % fullVocabFilePri)
    print('Secondary Vocab Model file %s' % fullVocabFileSec)
    torch.save(model.state_dict(), fullModelNamePth)  # Save model file as torch file
    fh = open(fullModelNamePck, 'wb')  # Save model file as pickle
    pickle.dump(model, fh)
    fh.close()
    fh = open(fullVocabFilePri, 'wb')
    pickle.dump(corpusEng.vocab_map, fh)  # Save primary vocab file as pickle
    fh.close()
    fh = open(fullVocabFileSec, 'wb')
    pickle.dump(corpusTr.vocab_map, fh)  # Save secondary vocab file as pickle
    fh.close()


# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(), torch.cuda.device_count()

###############################################################################
# Load data
###############################################################################
path = './'
wrdpath = '../wordvectors/'
compositionalMethod = 'additive'
fnamePri = 'english.1000.tok'
fnameSec = 'turkish.1000.tok'
corpusEng = Corpus(wrdpath, fnamePri)
corpusTr = Corpus(wrdpath, fnameSec)
corpusEng.data.cuda()
corpusTr.data.cuda()

vocab_size_pri = corpusEng.vocab_idx
vocab_size_sec = corpusTr.vocab_idx
embedding_dim = 64
batch_size = 100  # 100
max_epoch = 50  # 500
# learning_rate = 0.001
learning_rate_decay = 0.1
threshold = 10

momentlist = [0.2]  # ok
lrlist = [0.001]  # ok

ftestRes = open(wrdpath + compositionalMethod + "-cdlc-run.txt", "w")
ftestRes.write("name,epoch,lr,momentum,max_epoch,test score,total\n")

modelforCdlc = None

for mmt in momentlist:
    for lr in lrlist:
        lrStr = ("%2.15f" % lr).rstrip('0')
        mmtStr = ("%2.5f" % mmt).rstrip('0')
        prmStr = "%s-%s-%d-%d" % (lrStr, mmtStr, max_epoch, batch_size)
        fullModelNamePth = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStr + ".pth"
        fullModelNamePck = wrdpath + compositionalMethod + '-' + 'model1000EnTr-' + prmStr + ".pck"
        fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStr + ".vocab"
        fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStr + ".vocab"
        modelTrained = train_model(ftestRes, lr, mmt)
        dummy1, dummy2, testScore, total = test_model(modelTrained, corpusEng.data, corpusTr.data)
        ftestRes.write(
            "%s,%d,%s,%s,%d,%d,%d\n" % (compositionalMethod, 999, lrStr, mmtStr, max_epoch, testScore, total))
        ftestRes.flush()
        print("End Training")
#        saveModel(modelTrained)
#        print("Model Saved")

ftestRes.close()
print("End Program - Additive")


