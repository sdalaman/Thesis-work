
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

# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

def floatToStr(frm,prm):
    return (frm % prm).rstrip('0')

lrStrMdl = mres.floatToStr("%f",1e-3)
mmtStrMdl = mres.floatToStr("%f",0.2)
epochMdl = 500
batchSizeMdl = 100
embedding_dimMdl = 64
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)


wrdpath = '../wordvectors/1000/'
compositionalMethod = 'additive'
fullVocabFilePri = wrdpath + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )


classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
#classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

load = False

fResOut = open('cdlc-datafiles-list.txt', "w")
results  = []

for classname in classes:
    if load == False:
        print(datetime.datetime.today())
        positivePri, negativePri, maxLenPri = getData(data_pathPri,classname,vocabPri)
        print("%s loaded %s" % (classname,data_pathPri) )
        all_positivesPri=positivePri.size()[0]  # 122
        all_negativesPri=negativePri.size()[0]  # 118

        positiveSec, negativeSec, maxLenSec = getData(data_pathSec,classname,vocabSec)
        print("%s loaded %s" % (classname,data_pathSec) )
        all_positivesSec=positiveSec.size()[0]  # 27
        all_negativesSec=negativeSec.size()[0]  # 26
        res = " class : %s posPri %d negPri %d posSec %d negSec %d " % (classname,all_positivesPri,all_negativesPri,all_positivesSec,all_negativesSec)
        results.append(res)

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

fResOut.close()

print("End of classifier test")

