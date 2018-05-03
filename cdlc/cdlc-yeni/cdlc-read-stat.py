# Code in file nn/two_layer_net_nn.py
import torch
import torch.nn.init as init
from torch.autograd import Variable
import pickle
import numpy as np
import datetime
from datetime import datetime
import sys, getopt

class modelSet(object):
    def __init__(self):
        self.optimName = ""
        self.lr = 0
        self.momentum = 0
        self.initWeight = 0
        self.batchSize = 0
        self.maxEpoch = 0
        self.maxFold = 0
        self.threshold = 0
        self.hLayerNum = 0
        self.inputDims = 0
        self.hiddenDims = []
        self.outputDims = 0
        self.learningRateDecay = 0
        self.models = []
        self.accCl = {}

fname="cl-art-cdlc-accCl-2017-06-21-12-43-10.bin"

classinp = ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"h",["class=","fname="])
except getopt.GetoptError:
    print ('prog.py --fname=<fname> ')
    sys.exit(2)
for opt,arg in opts:
    if opt == '-h':
        print('prog.py --fname=<fname>')
        sys.exit()
    elif opt in ("--fname"):
        fname = arg

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
#classes = ['art']


accCl = {}
modelPrms = modelSet()
modelPrms = pickle.load(open(fname, 'rb'))
accCl = modelPrms.accCl

mn = 0
cnt = 0
avgCl = {}
avgMtd = {}

for cl in classes:
    avgCl[cl] = {}
    for mtd in accCl[cl]:
        avgCl[cl][mtd]={}
        for key in accCl[cl][mtd]:
            mn = 0
            cnt = 0
            for fld in accCl[cl][mtd][key]:
                if debugFlag == True:
                    print("class %s mtd %s fold : %d  acc : %.3f" % (cl,mtd,fld,accCl[cl][mtd][key][fld]))
                mn += accCl[cl][mtd][key][fld]
                cnt += 1
            avgCl[cl][mtd][key] = mn/cnt

print("----------")
print("----------")

for cl in avgCl:
    for mtd in avgCl[cl]:
        print("%s - %s : 10 fold avg acc (train/test) %.3f/%.3f " % (cl,mtd,avgCl[cl][mtd]["train"],avgCl[cl][mtd]["test"]))
print("----------")


i = 0