import os
import math
# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# import torch.optim as optim
import pickle
import datetime
import numpy as np


class args(object):
    pass


def getData(ln,data_path, classname):
    pos_fcnt = 0
    neg_fcnt = 0
    listpos = []
    listneg = []
    path = data_path + '/' + classname + '/positive'
    if os.path.exists(path):
        for file in os.listdir(path):
            with open(path + "/" + file, 'r') as f:
                listpos.append(file)
                text = f.read()
                words = text.split(" ")
                if len(words) > 0:
                    pos_fcnt += 1

    path = data_path + '/' + classname + '/negative'
    if os.path.exists(path):
        for file in os.listdir(path):
            with open(path + "/" + file, 'r') as f:
                listneg.append(file)
                text = f.read()
                words = text.split(" ")
                if len(words) > 0:
                    neg_fcnt += 1

    return pos_fcnt, neg_fcnt,listpos,listneg

#classes = ['biology','culture','science']
classesOld = ['art', 'arts', 'biology', 'business', 'creativity', 'culture', 'design', 'economics', 'education',
              'entertainment', 'health', 'politics', 'science', 'technology']
classes = ['art', 'arts', 'biology', 'business', 'creativity', 'culture', 'design', 'economics', 'education',
           'entertainment', 'global', 'health', 'politics', 'science', 'technology']
langList = [['en-tr','tr-en'],['en-de','de-en'],['en-fr','fr-en']]
dTypeList = ['tok','morph']
#dTypeList = ['tok']
trTstList = ['train','test']
data_path = "/home/saban/work/python/pytorch-works/additive/data/ted/"

load = False

fResOut = open('cdlc-datafiles-list.txt', "w")
results = []

for langSet in langList:
    for dType in dTypeList:
        for trTst in trTstList:
            data_pathPri = data_path + langSet[0] + "-" + dType + "/" + trTst + "/"
            data_pathSec = data_path + langSet[1] + "-" + dType + "/" + trTst + "/"
            results.append("\n\n")
            results.append(langSet[0].upper() + " " + dType.upper()+ " " + trTst.upper() +" data files stats")
            results.append("----------------------")
            for classname in classes:
                if load == False:
                    positivePriFCnt, negativePriFCnt,listPosPri,listNegPri = getData('en',data_pathPri, classname)
                    positiveSecFCnt, negativeSecFCnt,listPosSec,listNegSec = getData('tr',data_pathSec, classname)
                    res = " class : %s posPriFCnt %d negPriFCnt %d posSecFCnt %d negSecFCnt %d" % (
                    classname, positivePriFCnt, negativePriFCnt, positiveSecFCnt, negativeSecFCnt)
                    results.append(res)

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

fResOut.close()

print("End -------")
exit()


#####################################################

data_pathPri = "/home/saban/work/python/pytorch-works/additive/data/ted/en-tr-tok/train/"
data_pathSec = "/home/saban/work/python/pytorch-works/additive/data/ted/tr-en-tok/train/"
#/home/saban/work/python/pytorch-works/additive/data/ted/en-tr-tok/train/art/positive


results.append("\n\n")
results.append("TOK train data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriFCnt, negativePriFCnt,listPosPri,listNegPri = getData('en',data_pathPri, classname)
        positiveSecFCnt, negativeSecFCnt,listPosSec,listNegSec = getData('tr',data_pathSec, classname)
        res = " class : %s posPriFCnt %d negPriFCnt %d posSecFCnt %d negSecFCnt %d" % (
        classname, positivePriFCnt, negativePriFCnt, positiveSecFCnt, negativeSecFCnt)
        results.append(res)
        #print("%s",classname)
        #print(set(listPosPri) - set(listPosSec))
        #print(set(listPosSec) - set(listPosPri))
        #print(set(listNegPri) - set(listNegSec))
        #print(set(listNegSec) - set(listNegPri))


data_pathPri = "/home/saban/work/python/pytorch-works/additive/data/ted/en-tr-tok/test/"
data_pathSec = "/home/saban/work/python/pytorch-works/additive/data/ted/tr-en-tok/test/"

results.append("\n\n")
results.append("TOK test data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriFCnt, negativePriFCnt,listPosPri,listNegPri = getData('en',data_pathPri, classname)
        positiveSecFCnt, negativeSecFCnt,listPosSec,listNegSec = getData('tr',data_pathSec, classname)
        res = " class : %s posPriFCnt %d negPriFCnt %d posSecFCnt %d negSecFCnt %d" % (
        classname, positivePriFCnt, negativePriFCnt, positiveSecFCnt, negativeSecFCnt)
        results.append(res)
        #print("%s",classname)
        #print(set(listPosPri) - set(listPosSec))
        #print(set(listPosSec) - set(listPosPri))
        #print(set(listNegPri) - set(listNegSec))
        #print(set(listNegSec) - set(listNegPri))

data_pathPri = "/home/saban/work/python/pytorch-works/additive/data/ted/en-tr-morph/train/"
data_pathSec = "/home/saban/work/python/pytorch-works/additive/data/ted/tr-en-morph/train/"

results.append("\n\n")
results.append("morph train data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriFCnt, negativePriFCnt,listPosPri,listNegPri = getData('en',data_pathPri, classname)
        positiveSecFCnt, negativeSecFCnt,listPosSec,listNegSec = getData('tr',data_pathSec, classname)
        res = " class : %s posPriFCnt %d negPriFCnt %d posSecFCnt %d negSecFCnt %d" % (
        classname, positivePriFCnt, negativePriFCnt, positiveSecFCnt, negativeSecFCnt)
        results.append(res)
        #print("%s",classname)
        #print(set(listPosPri) - set(listPosSec))
        #print(set(listPosSec) - set(listPosPri))
        #print(set(listNegPri) - set(listNegSec))
        #print(set(listNegSec) - set(listNegPri))

data_pathPri = "/home/saban/work/python/pytorch-works/additive/data/ted/en-tr-morph/test/"
data_pathSec = "/home/saban/work/python/pytorch-works/additive/data/ted/tr-en-morph/test/"

results.append("\n\n")
results.append("morph test data files stats")
results.append("----------------------")

for classname in classes:
    if load == False:
        positivePriFCnt, negativePriFCnt,listPosPri,listNegPri = getData('en',data_pathPri, classname)
        positiveSecFCnt, negativeSecFCnt,listPosSec,listNegSec = getData('tr',data_pathSec, classname)
        res = " class : %s posPriFCnt %d negPriFCnt %d posSecFCnt %d negSecFCnt %d" % (
        classname, positivePriFCnt, negativePriFCnt, positiveSecFCnt, negativeSecFCnt)
        results.append(res)
        #print("%s",classname)
        #print(set(listPosPri) - set(listPosSec))
        #print(set(listPosSec) - set(listPosPri))
        #print(set(listNegPri) - set(listNegSec))
        #print(set(listNegSec) - set(listNegPri))

for res in results:
    print(res)
    fResOut.write(res)
    fResOut.write("\n")

fResOut.close()

print("End -------")
