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
        self.regFactor = 0
        self.regType = ""
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
        self.models = {}
        self.runStats = {}

class lossSet(object):
    def __init__(self):
        self.epoch = []
        self.loss = []
        self.mean = []

class foldStat(object):
    def __init__(self):
        self.prec = self.recall = self.f1score = self.acc = 0
        self.correct = self.total = self.tp = self.pp = self.ap = self.an = self.fp = 0

def selectOptimizer(model, mdlPrms):
        # "SGD","RMSprop","Adadelta","Adagrad","Adam","Adamax","ASGD"
        if mdlPrms.optimName == "SGD":
            if mdlPrms.momentum == 0:
                nesterov = False
            else:
                nesterov = True
            optimizer = torch.optim.SGD(model.parameters(), lr=mdlPrms.lr, momentum=mdlPrms.momentum, dampening=0,
                                        weight_decay=0, nesterov=nesterov)
        elif mdlPrms.optimName == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=mdlPrms.lr, alpha=0.99, eps=1e-08, weight_decay=0,
                                            momentum=mdlPrms.momentum, centered=False)
        elif mdlPrms.optimName == "Adadelta":
            optimizer = torch.optim.Adadelta(model.parameters(), lr=mdlPrms.lr, rho=0.9, eps=1e-06, weight_decay=0)
        elif mdlPrms.optimName == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=mdlPrms.lr, lr_decay=0, weight_decay=0)
        elif mdlPrms.optimName == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=mdlPrms.lr, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif mdlPrms.optimName == "Adamax":
            optimizer = torch.optim.Adamax(model.parameters(), lr=mdlPrms.lr, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0)
        elif mdlPrms.optimName == "ASGD":
            optimizer = torch.optim.ASGD(model.parameters(), lr=mdlPrms.lr, lambd=0.0001, alpha=0.75, t0=1000000.0,
                                         weight_decay=0)
        else:  # default optmizer SGD
            if mdlPrms.momentum == 0:
                nesterov = False
            else:
                nesterov = True
            optimizer = torch.optim.SGD(model.parameters(), lr=mdlPrms.lr, momentum=mdlPrms.momentum, dampening=0,
                                        weight_decay=0, nesterov=nesterov)
        return optimizer

def calcStats(pred,actual):
    tmpStat = foldStat()
    tmpStat.total = pred.size()[0]
    for i in range(pred.size()[0]):
        if  pred[i][0] == 1:
            tmpStat.pp += 1
        if  actual[i] == 1:
            tmpStat.ap += 1
            if pred[i][0] == 1:
                tmpStat.tp += 1
        else:
            tmpStat.an += 1
            if pred[i][0] == 1:
                tmpStat.fp += 1

        if pred[i][0] == actual[i]:
            tmpStat.correct += 1

    tmpStat.acc = tmpStat.correct / tmpStat.total
    if tmpStat.pp != 0:
        tmpStat.prec = tmpStat.tp/tmpStat.pp
    if tmpStat.ap != 0:
        tmpStat.recall = tmpStat.tp / tmpStat.ap
    if (tmpStat.prec+tmpStat.recall) != 0:
        tmpStat.f1score = 2*tmpStat.prec*tmpStat.recall / (tmpStat.prec+tmpStat.recall)
    return tmpStat

def calcFoldStats(runStats,pFlag):
    avgCl = {}
    for cl in classes:
        avgCl[cl] = {}
        for mtd in runStats[cl]:
            avgCl[cl][mtd] = {}
            for runPrm in runStats[cl][mtd]:
                avgCl[cl][mtd][runPrm] = {}
                for key in runStats[cl][mtd][runPrm]:
                    mn = 0
                    cnt = 0
                    for fld in runStats[cl][mtd][runPrm][key]:
                        sts = runStats[cl][mtd][runPrm][key][fld]
                        if pFlag == True:
                            print("class %s mtd %s %s (%s)" % (cl, mtd,runPrm, key))
                            print(
                                "        fold %d - prec : %.3f recall : %.3f f1score : %.3f acc : %d/%d = %.3f p:%%%.3f n:%%%.3f" %
                                (fld+1, sts.prec, sts.recall, sts.f1score, sts.correct, sts.total, sts.acc,
                                100 * sts.ap / (sts.ap + sts.an), 100 * sts.an / (sts.ap + sts.an)))
                        mn += sts.acc
                        cnt += 1
                    avgCl[cl][mtd][runPrm][key] = mn / cnt
                    print("class %s mtd %s %s (%s) %f ---------" % (cl, mtd, runPrm, key,mn / cnt))
    return avgCl

def printClassStats(avgCl):
    print("----------")
    print("parameters")
    summ = ("classnames %s - maxFold %d - maxEpoch %d hidden %d - lr %f - momentum %f - regularization %s - regfactor %f" % (clName,mPrms.maxFold, mPrms.maxEpoch,mPrms.hiddenDims,mPrms.lr, mPrms.momentum, mPrms.regType, mPrms.regFactor))
    print(summ)
    for cl in avgCl:
        print("%s - %s : %d fold avg acc (train/test) %.3f/%.3f " % (
        cl, mtd, mPrms.maxFold, avgCl[cl][mtd]["train"], avgCl[cl][mtd]["test"]))
        summ += " %s (train/test) %.3f/%.3f" % (mtd,avgCl[cl][mtd]["train"], avgCl[cl][mtd]["test"])
    print("----------")
    return summ


def shuffleData(xTr,yTr):
    shuffle = np.random.permutation(xTr.size()[0])
    x = xTr.clone()
    y = yTr.clone()
    for i in range(xTrain.size()[0]):
        x[i] = xTr[shuffle[i]]
        y[i] = yTr[shuffle[i]]
    return x,y


def trainModel(mPrms,x,y):

    modelTr = torch.nn.Sequential(
        torch.nn.Linear(mPrms.inputDims, mPrms.hiddenDims),
        torch.nn.Linear(mPrms.hiddenDims, mPrms.outputDims),
        torch.nn.Sigmoid()
    ).cuda()

    # modelMLP = MLP().cuda()

    optimizer = selectOptimizer(modelTr, mPrms)
    if mPrms.regType == "L1":
        regl = l1_loss
    else:
        regl = l2_loss

    for param in modelTr.parameters():
        init.uniform(param, -1 * mPrms.initWeight, mPrms.initWeight)

    W = np.random.randn(mPrms.inputDims, mPrms.hiddenDims)
    u, s, v = np.linalg.svd(W)
    uu = torch.FloatTensor(u).cuda()
    #modelTr[0].weight.data = uu[0:W.shape[1],0:W.shape[0]]

    #a = b = 0
    #for i in range(xTrain.size()[0]):
    #    a += torch.sum(x[i] - xTrain[shuffle[i]])
    #    b += (y[i] - yTrain[shuffle[i]])

    #if (a != 0 or b != 0):
    #    exit(2)

    x = Variable(x.cuda())
    y = Variable(y.cuda())

    lossValues = lossSet()

    for t in range(mPrms.maxEpoch):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Variable of input data to the Module and it produces
        # a Variable of output data.
        y_pred = modelTr(x)

        # Compute and print loss. We pass Variables containing the predicted and true
        # values of y, and the loss function returns a Variable containing the loss.
        loss = loss_fn(y_pred, y) + regl(regFactor, modelTr)

        lossValues.epoch.append(t)
        lossValues.loss.append(loss.data[0])
        # lossValues.mean.append(np.mean(lossValues.loss))

        if debugFlag == True:
            print("epoch %d lr %2.10f - %.3f " % (t + 1, optimizer.param_groups[0]["lr"], loss.data[0]))

            # Zero the gradients before running the backward pass.
            modelTr.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Variables with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Variable, so
        # we can access its data and gradients like we did before.

        optimizer.step()

        # for param in model.parameters():
        #    param.data -= learning_rate * param.grad.data

        # learning_rate = lr * (1 ** (t // 50))
        if ((t + 1) % 100) == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * mPrms.learningRateDecay
            la = 0

    return modelTr,lossValues


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
#N, D_in,H, D_out = 240, 128,128, 1

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.BCELoss(size_average=False)
l1_crit = torch.nn.L1Loss(size_average=True)


def l1_loss(factor,model):
    reg_loss = 0
    for param in model.parameters():
        target = Variable(param.data.clone().fill_(0))
        reg_loss += l1_crit(param,target)
    return factor * reg_loss

def l2_loss(factor,model):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(torch.pow(param.data, 2))
    return factor * reg_loss


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.out1 = torch.nn.Linear(D_in,D_out)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.out1(x)
        x = self.sig(x)
        return x


allPri = {"additive": {}, "tanh": {}}
allSec = {"additive": {}, "tanh": {}}
targetsPri = {}
targetsSec = {}

#cmpMethodWord2Vec = "additive"
#classname = "art"

#default values
learning_rate = 1e-4 # 1e-4
momentum = 0.5
regFactor = 10
regType = "L1"
maxFold = 10
maxEpoch = 300
dout=0

modelPrms = modelSet()

classinp = ""
try:
    opts, args = getopt.getopt(sys.argv[1:],"h",["class=","maxepoch=","maxfold=","lr=","dout=","momentum=","lfactor=","ltype="])
except getopt.GetoptError:
    print ('prog.py --class=<classname> --maxepoch=<maxepoch> --maxfold=<maxfold> --lr=<lr> --momentum=<mmt> --lfactor=<> --ltype=<L1/L2> --dout=<0/1>')
    sys.exit(2)
for opt,arg in opts:
    if opt == '-h':
        print('prog.py --class <classname> --maxepoch <maxepoch> --maxfold=<maxfold> --lr=<lr> --momentum=<mmt> --lfactor=<> --ltype=<L1/L2> --dout=<0/1>')
        sys.exit()
    elif opt in ("--class"):
        classinp = arg
    elif opt in ("--maxepoch"):
        maxEpoch = int(arg)
    elif opt in ("--maxfold"):
        maxFold = int(arg)
    elif opt in ("--lr"):
        learning_rate = float(arg)
    elif opt in ("--momentum"):
        momentum = float(arg)
    elif opt in ("--lfactor"):
        regFactor = float(arg)
    elif opt in ("--ltype"):
        regType = arg
    elif opt in ("--dout"):
        dout = int(arg)

modelPrms.optimName = "RMSprop"
modelPrms.lr = learning_rate
modelPrms.momentum = momentum
modelPrms.initWeight = 0.001 #0.0234375
modelPrms.batchSize = 0
modelPrms.maxEpoch = maxEpoch
modelPrms.maxFold = maxFold
modelPrms.regFactor = regFactor
modelPrms.regType = regType
modelPrms.threshold = 0
modelPrms.hLayerNum = 1
modelPrms.inputDims = 128
modelPrms.hiddenDims = 128
modelPrms.outputDims = 1
modelPrms.learningRateDecay = 1


runStats = {}
accFold = {"train" : {} , "test" : {}}
avgCl = {}


#x = Variable(torch.randn(N, D_in))
#y = Variable(torch.round(torch.rand(N, D_out)))
#y = Variable(torch.round(torch.rand(N, D_out)), requires_grad=False)

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
#classes = ['art']
if dout == 0:
    debugFlag = False
else:
    debugFlag = True

path="../"

classes = []
if classinp != "":
    classes.append(classinp)
else:
    classes.append('art')  # default class value

clsList = "cl"
summary =  []

print("Begin : %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

prmList= []
prmList.append([40,1000])
prmList.append([40,5000])
prmList.append([40,10000])


for classname in classes:

    clsList =clsList +"-"+ classname

    runStats[classname] = {}
    runStats[classname] = {}
    modelPrms.models[classname] = {}

    #for cmpMethodWord2Vec in ["additive"]:  #["additive","tanh"]:
    for cmpMethodWord2Vec in ["additive","tanh"]:

        runStats[classname][cmpMethodWord2Vec] = {}
        runStats[classname][cmpMethodWord2Vec] = {}
        modelPrms.models[classname][cmpMethodWord2Vec] = {}

        allPri[cmpMethodWord2Vec]["minmax"] = pickle.load(
            open(path+classname + "-" + cmpMethodWord2Vec + "-model-allPri-minmax-mlp-simple.pck", 'rb'))
        allSec[cmpMethodWord2Vec]["minmax"] = pickle.load(
            open(path+classname + "-" + cmpMethodWord2Vec + "-model-allSec-minmax-mlp-simple.pck", 'rb'))

        #allPri[cmpMethodWord2Vec]["additive"] = pickle.load(
        #open(path + classname + "-" + cmpMethodWord2Vec + "-model-allPri-additive-mlp-simple.pck", 'rb'))
        #allSec[cmpMethodWord2Vec]["additive"] = pickle.load(
        #open(path + classname + "-" + cmpMethodWord2Vec + "-model-allSec-additive-mlp-simple.pck", 'rb'))

        targetsPri[cmpMethodWord2Vec] = pickle.load(
            open(path+classname + "-" + cmpMethodWord2Vec + "-model-targetsPri-mlp-simple.pck", 'rb'))
        targetsSec[cmpMethodWord2Vec] = pickle.load(
            open(path+classname + "-" + cmpMethodWord2Vec + "-model-targetsSec-mlp-simple.pck", 'rb'))

        xTrain = allPri[cmpMethodWord2Vec]["minmax"].data
        yTrain = targetsPri[cmpMethodWord2Vec]

        xTest = allSec[cmpMethodWord2Vec]["minmax"].data
        yTest = targetsSec[cmpMethodWord2Vec]

        foldStats = {"train": {}, "test": {}}

        models = []



        for prms in prmList:
            modelPrms.hiddenDims = prms[0]
            modelPrms.maxEpoch = prms[1]
            prmStr = "H:%d-E:%d" % (modelPrms.hiddenDims,modelPrms.maxEpoch)
            runStats[classname][cmpMethodWord2Vec][prmStr] = {}
            runStats[classname][cmpMethodWord2Vec][prmStr] = {}

            print("parameters")
            print("classnames %s" % (classes))
            print("maxFold %d maxEpoch %d" % (modelPrms.maxFold, modelPrms.maxEpoch))
            print("learning_rate %f momentum %f reg %s regfactor %f hidden dim %d" % (modelPrms.lr, modelPrms.momentum,modelPrms.regType,modelPrms.regFactor,modelPrms.hiddenDims))

            for fold in list(range(modelPrms.maxFold)):
                print("classname %s method %s - fold %d" % (classname, cmpMethodWord2Vec, fold + 1))
                print("%s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                xTr, yTr = shuffleData(xTrain, yTrain)
                #xTr = xTrT[0:len,].clone()
                #yTr = yTrT[0:len, ].clone()
                #for i in range(4):
                #    xTrT, yTrT = shuffleData(xTrain, yTrain)
                #    xTr = torch.cat((xTr,xTrT[0:len,]),0)
                #    yTr = torch.cat((yTr, yTrT[0:len, ]),0)

                modelMLP,lossValues = trainModel(modelPrms, xTr, yTr)

                yTr_pred = modelMLP( Variable(xTr.cuda()))
                trStat = calcStats(torch.round(yTr_pred.data),yTr)
                foldStats["train"][fold] =  trStat

                xTst = xTest.clone()
                yTst = yTest.clone()
                xTst = Variable(xTst.cuda())
                yTst = Variable(yTst.cuda())
                yTst_pred_test = modelMLP(xTst)
                #accFold["test"][fold] = torch.sum(torch.round(y_pred_test.data) == y.data) / y.data.size()[0]
                #tstStat = foldStat()
                tstStat = calcStats(torch.round(yTst_pred_test.data), yTst.data)
                foldStats["test"][fold] = tstStat
                models.append(modelMLP.state_dict())

                #fname = "cdlc-lossValues-fold-" + str(fold) + "-" + classname + "-" + cmpMethodWord2Vec + "-"\
                #        + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".bin"
                #fh = open(fname, 'wb')  # Save model file as pickle
                #pickle.dump(lossValues, fh)
                #fh.close()

            modelPrms.models[classname][cmpMethodWord2Vec][prmStr]=models
            runStats[classname][cmpMethodWord2Vec][prmStr]["train"]=foldStats["train"]
            runStats[classname][cmpMethodWord2Vec][prmStr]["test"]=foldStats["test"]
            print("----------")

debugFlag = True
avgCl = calcFoldStats(runStats,debugFlag)

print("End : %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for cl in avgCl:
    for mtd in avgCl[cl]:
        for prm in avgCl[cl][mtd]:
            str = cl + " " + mtd +" " + prm + " (train/test):" + "%.3f"%(avgCl[cl][mtd][prm]["train"]) +"/" + "%.3f"%(avgCl[cl][mtd][prm]["test"])
            print(str)

#print("----------")
#avgCl = foldStats(classes,runStats,debugFlag)
#printClassStats(avgCl,modelPrms)

#modelPrms.runStats = runStats
#fname=clsList+"-cdlc-runStats-"+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+".bin"
#fh = open(fname, 'wb')  # Save model file as pickle
#pickle.dump(modelPrms, fh)
#fh.close()
#print("%s saved" % fname)

i = 0