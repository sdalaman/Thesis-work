
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



class args(object):
    pass
    
class LossValues(object):
    def __init__(self):
        self.x = []
        self.y1 = []
        self.y2 = []
        self.mean = []


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



class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,num_classes,lr,momentum,maxSent):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.nlayers = num_layers
        self.lr = lr
        self.momentum = momentum
        self.maxSent = maxSent
        self.build_model()
    # end constructor

    def build_model(self):
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size,nonlinearity='relu').cuda()
        self.fc = nn.Linear(self.hidden_size, self.num_classes).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_prm)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,momentum=self.momentum, centered=False)
        #self.hidden = autograd.Variable(torch.randn(1, self.hidden_size).cuda())

    def init_hidden(self,initrange):
        for ww in self.parameters():
            init.uniform(ww.data, -1 * initrange, initrange)
        weight = next(self.parameters()).data
        return autograd.Variable(weight.new(1,self.hidden_size).zero_().cuda())
    # end method build_model

    def forwardOld(self, x,hidden):
        for i in range(x.data.size()[0]):
            inp = autograd.Variable(x.data[i].resize_(1,x.data[i].size()[0]).cuda())
            hidden = self.rnn(inp,hidden)
        out = self.fc(hidden)
        return out,hidden
    # end method forward

    def forward(self, x,hidden):
        for i in range(x.data.size()[0]):
        #for i in range(self.maxSent):
            inp = x[i]
            hidden = self.rnn(inp,hidden)
        out = self.fc(hidden)
        return out,hidden
    # end method forward

    def forwardLSTM(self, x,hidden):
        for i in range(x.data.size()[0]):
        #for i in range(self.maxSent):
            inp = x[i]
            hidden ,cx = self.rnn(inp,hidden)
        out = self.fc(hidden)
        return out,hidden,cx


    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == autograd.Variable:
            return autograd.Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)
# end class RNNClassifier

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
    file_sent_list = {}
    for fl in data:
        sentences = []
        for sent in data[fl]:
            x = torch.Tensor(longest_sent)
            newSent = padding(sent, longest_sent)
            for idx in range(longest_sent):
                if vocab.get(newSent[idx]) != None:
                    x[idx] = vocab[newSent[idx]]
                else:
                    x[idx] = 0
            sentences.append(x)
        file_sent_list[fl] = sentences
    return file_sent_list

def getData(data_path,classname,vocab):
    pos_data = {}
    neg_data = {}
    sentences = []
    longest_sent = 0
    path = data_path+'/'+classname+'/positive'
    for file in os.listdir(path):
        sentences = []
        with open(path+"/"+file, 'r') as f:
            for line in f:
                for snt in line.split("."):
                    if not snt == '\n':
                        sentences.append(snt.split())
                        snt_len = len(snt.split())
                        if snt_len >  longest_sent:
                            longest_sent = snt_len;
        pos_data[file]= sentences

    path = data_path+'/'+classname+'/negative'
    for file in os.listdir(path):
        sentences = []
        with open(path+"/"+file, 'r') as f:
            for line in f:
                for snt in line.split("."):
                    if not snt == '\n':
                        sentences.append(snt.split())
                        snt_len = len(snt.split())
                        if snt_len >  longest_sent:
                            longest_sent = snt_len
        neg_data[file]=sentences

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

def trainClassifier(allDataTrain,allDataTest,learning_rate,momentum,max_epoch,saveModel,resFile,maxSent):
    classifier = RNNClassifier(embedding_dim,embedding_dim,2,2,learning_rate,momentum,max_sent).cuda()
    batch_size=1
    weight_clip=0.25
    hidden = classifier.init_hidden(init_weight)
    #for param in classifier.parameters():
        #init.uniform(param, -1 * 0.0000001, 0.0000001)
    #    init.uniform(param, -1 * 0.0000001, 0.0000001)
    #loss_function = nn.BCELoss(size_average=True).cuda()
    #optimizer = optim.Adadelta(classifier.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(classifier.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0,momentum=momentum, centered=False)
    #optimizer = optim.LBFGS(classifier.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(classifier.parameters(), lr= learning_rate, momentum=n_momentum, dampening=0, weight_decay=1e-4, nesterov=True)
    errors = []
    lrStr = ("%2.15f" % learning_rate).rstrip('0')
    momentumStr = ("%2.15f" % momentum).rstrip('0')
    lossVal = LossValues()
    epc = 0

#   slist.append(modelLoaded.embeddings_pri(autograd.Variable(sentence.long().cuda())))
    posValues = list(allDataTrain[1].values())
    negValues = list(allDataTrain[0].values())
    all_positives=len(posValues)
    all_negatives=len(negValues)
    totFileCount = all_positives + all_negatives

    for fold in range(folds):
        errors = []
        for epoch in range(max_epoch):
            print("class :  %s fold %d epoch %d" % (classname,fold,epoch))
            epc += 1
            shuffle = np.random.permutation(totFileCount)
            lr=classifier.optimizer.param_groups[0]['lr']
            lrStr = ("%2.15f" % lr).rstrip('0')
            for i in range(math.floor(totFileCount*train_per)):
                if shuffle[i] < all_positives:
                    docInp = posValues[shuffle[i]].data.clone()
                    target = autograd.Variable(torch.Tensor(1).fill_(1).long().cuda(), requires_grad=False)
                else:
                    docInp = negValues[shuffle[i]-all_positives].data.clone()
                    target = autograd.Variable(torch.Tensor(1).fill_(0).long().cuda(), requires_grad=False)

                docDiv = math.floor(docInp.size()[0] / maxSent)
                if docInp.size()[0] % maxSent != 0:
                    docDiv += 1
                for sCnt in range(docDiv):
                    #inp = docInp[sCnt:sCnt+max_sent+1]
                    inp = docInp[sCnt*maxSent:(sCnt + 1)*maxSent]
                    inp = autograd.Variable(inp.resize_(inp.size()[0],1,inp.size()[1]).cuda())
                    hidden = classifier.repackage_hidden(hidden)
                    classifier.zero_grad()
                    pred,hidden = classifier.forward(inp,hidden)
                    loss = classifier.criterion(pred,target)
                    print("sent count %d " %  inp.data.size()[0])
                    print("fold %d epoch %d lr %s mmt %s sntc %d- pred %f/%f target %f loss %f " % (fold,epoch,lrStr,momentumStr,maxSent,pred.data[0][0],pred.data[0][1],target.data[0], loss.data[0]))
                    loss.backward()
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    #torch.nn.utils.clip_grad_norm(classifier.parameters(), weight_clip)
                    classifier.optimizer.step()
                    #for p in classifier.parameters():
                    #    print(p.grad.data)
                    errors.append(loss.data[0])
                    lossVal.y1.append(loss.data[0])
                    mean = torch.mean(torch.Tensor(errors))
                    lossVal.mean.append(mean)

                #adjust_learning_rate(optimizer, epc, threshold, learning_rate, learning_rate_decay)
        #print("lr decayed %f" % get_learning_rate(optimizer))
        correct, predicted_positives, predicted_negatives, true_positives, true_negatives,test_all_pos,test_all_neg = testClassifier(classifier, allDataTest,maxSent)
        calculateAndPrintScores(resFile,fold,lrStr,momentum,maxSent,correct,predicted_positives,predicted_negatives,true_positives,true_negatives,test_all_pos,test_all_neg)

    lossVal.x = range(folds*max_epoch*math.floor(totFileCount*train_per))

    if saveModel == True:
        lrStr = ("%2.15f" % learning_rate).rstrip('0')
        fname = "%s%s-rnn-cdlc-loss-values-%s-%s-%d-%d.bin" % (path,compositionalMethod,lrStr,momentumStr,maxSent,max_epoch)
        fh = open(fname, 'wb')  # Save model file as pickle
        pickle.dump(lossVal, fh)
        fh.close()
    return classifier


def calculateAndPrintScores(outFile,fold,lrStr,momentumStr,maxSent,correct,predicted_positives,predicted_negatives,true_positives,true_negatives,all_positives,all_negatives):
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    score = 0.0
    accuracy = 0.0
    error_rate = 0.0
    if predicted_positives != 0 and all_positives != 0:
        precision = true_positives / predicted_positives
        recall = true_positives / all_positives
        accuracy = (true_positives + true_negatives) / (all_positives + all_negatives)
        f1_score = (2 * precision * recall / (precision + recall))
        error_rate = (all_positives + all_negatives - true_positives - true_negatives) / (all_positives + all_negatives)
        score = correct / (all_positives + all_negatives)
        outFile.write("cdlc,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,\n" %
                      (fold, lrStr, momentumStr, max_epoch, maxSent,all_positives, all_negatives, correct, predicted_positives,
                       predicted_negatives, true_positives, true_negatives, precision, recall, f1_score, score,accuracy,error_rate))


def testClassifier(classifier, allData,maxSent):
    correct=0
    predicted_positives=0
    predicted_negatives=0
    true_positives=0
    true_negatives=0

    posValues = list(allData[1].values())
    negValues = list(allData[0].values())
    all_positives=len(posValues)
    all_negatives=len(negValues)

    hidden = classifier.init_hidden(init_weight)
    target = 1

    for i in range(all_positives):
        x = posValues[i].data.clone()
        #x = autograd.Variable(x.resize_(x.size()[0], 1, x.size()[1]).cuda())

        docDiv = math.floor(x.size()[0] / maxSent)
        if x.size()[0] % maxSent != 0:
            docDiv += 1

        posCnt = negCnt = 0
        for sCnt in range(docDiv):
            inp = x[sCnt * maxSent:(sCnt + 1) * maxSent]
            inp = autograd.Variable(inp.resize_(inp.size()[0], 1, inp.size()[1]).cuda())
            hidden = classifier.repackage_hidden(hidden)
            pred,hidden = classifier.forward(inp,hidden)
            _,predicted = torch.max(pred.data, 1)
            if predicted[0][0]  == 0:
                negCnt += 1
            else:
                posCnt += 1

        if posCnt <= negCnt:
            output = 0
            predicted_negatives += 1
        else:
            output = 1
            predicted_positives += 1
            correct = correct + 1
            true_positives += 1

        print("predicted : %f - target : %d - output : %d" % (pred.data[0][0], target,output))

    target = 0
    for i in range(all_negatives):
        x = negValues[i].data.clone()
        #x = autograd.Variable(x.resize_(x.size()[0], 1, x.size()[1]).cuda())

        docDiv = math.floor(x.size()[0] / maxSent)
        if x.size()[0] % maxSent != 0:
            docDiv += 1

        posCnt = negCnt = 0
        for sCnt in range(docDiv):
            inp = x[sCnt * maxSent:(sCnt + 1) * maxSent]
            inp = autograd.Variable(inp.resize_(inp.size()[0], 1, inp.size()[1]).cuda())
            hidden = classifier.repackage_hidden(hidden)
            pred,hidden = classifier.forward(inp,hidden)
            _,predicted = torch.max(pred.data, 1)
            if predicted[0][0]  == 0:
                negCnt += 1
            else:
                posCnt += 1

        if negCnt > posCnt:
            output = 0
            predicted_negatives += 1
            correct = correct + 1
            true_negatives += 1
        else:
            output = 1
            predicted_positives += 1

        print("predicted : %f - target : %d - output : %d" % (pred.data[0][0], target,output))

    print("all_pos : %d all_neg : %d " % (all_positives,all_negatives))
    print("correct : %d pred_pos : %d " % (correct,predicted_positives))
    print("true_pos : %d  true_neg : %d" % (true_positives,true_negatives))

    return correct,predicted_positives,predicted_negatives,true_positives,true_negatives,all_positives,all_negatives


def testClassifierOld(classifier, allData):
    correct=0
    predicted_positives=0
    predicted_negatives=0
    true_positives=0
    true_negatives=0

    posValues = list(allData[1].values())
    negValues = list(allData[0].values())
    all_positives=len(posValues)
    all_negatives=len(negValues)

    hidden = classifier.init_hidden(init_weight)
    target = 1

    for i in range(all_positives):
        hidden = classifier.repackage_hidden(hidden)
        x = posValues[i].data.clone()
        x = autograd.Variable(x.resize_(x.size()[0], 1, x.size()[1]).cuda())
        #x = autograd.Variable(x.cuda())
        pred,hidden = classifier.forward(x,hidden)
        _,predicted = torch.max(pred.data, 1)
        if predicted[0][0]  == 0:
            output = 0
            predicted_negatives += 1
        else:
            output = 1
            predicted_positives += 1
            correct = correct + 1
            true_positives += 1
        print("predicted : %f - target : %d - output : %d" % (pred.data[0][0], target,output))

    target = 0
    for i in range(all_negatives):
        hidden = classifier.repackage_hidden(hidden)
        x = negValues[i].data.clone()
        x = autograd.Variable(x.resize_(x.size()[0], 1, x.size()[1]).cuda())
        pred,hidden = classifier.forward(x,hidden)
        _,predicted = torch.max(pred.data, 1)
        if predicted[0][0]  == 0:
            output = 0
            predicted_negatives += 1
            correct = correct + 1
            true_negatives += 1
        else:
            output = 1
            predicted_positives += 1
        print("predicted : %f - target : %d - output : %d" % (pred.data[0][0], target,output))

    print("all_pos : %d all_neg : %d " % (all_positives,all_negatives))
    print("correct : %d pred_pos : %d " % (correct,predicted_positives))
    print("true_pos : %d  true_neg : %d" % (true_positives,true_negatives))

    return correct,predicted_positives,predicted_negatives,true_positives,true_negatives,all_positives,all_negatives


def saveClassifierModel(model,prmStr):
    fnamePth = path + compositionalMethod + '-' + 'classifierModel1000EnTr'+prmStr+".pth"
    fnamePck = path + compositionalMethod + '-' + 'classifierModel1000EnTr' + prmStr + ".pck"
    print('Classifier Model file pth %s' % fnamePth)
    print('Classifier Model file pck %s' % fnamePck)
    torch.save(model.state_dict(), fnamePth)   # Save model file as torch file
    fh = open(fnamePck, 'wb')  # Save model file as pickle
    pickle.dump(model, fh)
    fh.close()

def prepareSentenceEmbeddings(all,positive,negative,modelForward):
    allPos = {}
    for fl in positive:
        row = len(positive[fl])
        col = len(positive[fl][0])
        sTensor = torch.zeros(row,col).long().cuda()
        rcnt = 0
        for sentence in positive[fl]:
            sTensor[rcnt] = torch.Tensor(sentence).long().cuda()
            rcnt += 1
        allPos[fl] = modelForward(autograd.Variable(sTensor))
    all[1] = allPos
    print("pos " + str(datetime.datetime.today()))
    allNeg = {}
    for fl in negative:
        row = len(negative[fl])
        col = len(negative[fl][0])
        sTensor = torch.Tensor(row,col).long().cuda()
        rcnt = 0
        for sentence in negative[fl]:
            sTensor[rcnt] = torch.Tensor(sentence).long().cuda()
            rcnt += 1
        allNeg[fl] = modelForward(autograd.Variable(sTensor))
    all[0] = allNeg
    print("neg " + str(datetime.datetime.today()))
    return all


# Set the random seed manually for reproducibility.
torch.manual_seed(1111)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1111)
torch.cuda.is_available(),torch.cuda.device_count()

path = './'
wrdpath = '../wordvectors/'
compositionalMethod = 'additive'
lrStrMdl = ("%f" % 1e-3).rstrip('0')
mmtStrMdl = ("%f" % 0.2).rstrip('0')
epochMdl = 100
batchSizeMdl = 100
prmStrMdl = "%s-%s-%d-%d" % (lrStrMdl, mmtStrMdl, epochMdl, batchSizeMdl)
fullModelNamePck = wrdpath = '../wordvectors/' + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pck'
fullModelNamePth = wrdpath = '../wordvectors/' + compositionalMethod + '-' + 'model1000EnTr-' + prmStrMdl + '.pth'
fullVocabFilePri = wrdpath = '../wordvectors/' + compositionalMethod + '-' + 'english.1000EnTr-' + prmStrMdl + '.vocab'
fullVocabFileSec = wrdpath = '../wordvectors/' + compositionalMethod + '-' + 'turkish.1000EnTr-' + prmStrMdl + '.vocab'
embedding_dim = 64
batch_size = 100
max_epoch = 5 # 300
learning_rate_decay = 0.01
threshold = 100
folds = 1 # 10
test_per = 0.25
train_per = 1

#classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology']
classes = ['art']
data_pathPri = "/home/saban/work/additive/data/cdlc_en_tr/english"
data_pathSec = "/home/saban/work/additive/data/cdlc_en_tr/turkish"

print('Model file %s' % fullModelNamePth)
print('Primary vocab file %s' % fullVocabFilePri)
print('Secondary vocab file %s' % fullVocabFileSec)
vocabPri = pickle.load( open( fullVocabFilePri, "rb" ) )
vocabSec = pickle.load( open( fullVocabFileSec, "rb" ) )

#modelLoadedPth = BiLingual(len(vocabPri),len(vocabSec),embedding_dim).cuda()
#modelLoadedPth.init_weights()
#modelLoadedPth.load_state_dict(torch.load(fullModelNamePth))

modelLoadedPck = BiLingual(len(vocabPri),len(vocabSec),embedding_dim).cuda()
modelLoadedPck.init_weights()
modelLoadedPck= pickle.load( open( fullModelNamePck, "rb" ) )

modelLoaded = modelLoadedPck

ftestRes = open(path + compositionalMethod + '-' + 'rnn-cdlc-run.txt', "w")
ftestRes.write("name,fold,lr,momentum,max_epoch,max_sent,all_positives,all_negatives,correct,predicted_positives,predicted_negatives,"
               "true_positives,true_negatives,precision,recall,f1_score,score,accuracy,error_rate\n")

for classname in classes:
    save = False
    load = True
    fnamePri = path + 'allPri' + '.pck'
    fnameSec = path + 'allSec' + '.pck'
    allPri = {}
    allSec = {}

    if save == True:
        print(datetime.datetime.today())
        positivePri, negativePri = getData(data_pathPri,classname,vocabPri)
        print("%s loaded %s" % (classname,data_pathPri) )

        allPri = {}
        allPri = prepareSentenceEmbeddings(allPri,positivePri,negativePri,modelLoaded.forwardPri)

        print(datetime.datetime.today())

        positiveSec, negativeSec= getData(data_pathSec,classname,vocabSec)
        positiveSecT = {}
        negativeSecT = {}
        tot = math.floor(len(positiveSec)*test_per)
        cnt = 0
        for key in positiveSec:
            positiveSecT[key] = positiveSec[key]
            cnt += 1
            if cnt == tot:
                break
        tot = math.floor(len(negativeSec)*test_per)
        cnt = 0
        for key in negativeSec:
            negativeSecT[key] = negativeSec[key]
            cnt += 1
            if cnt == tot:
                break

        positiveSec = positiveSecT
        negativeSec = negativeSecT

        print("%s loaded %s" % (classname,data_pathSec) )

        allSec = {}
        allSec = prepareSentenceEmbeddings(allSec,positiveSec,negativeSec,modelLoaded.forwardSec)

        print(datetime.datetime.today())

        positivePri = {}
        negativePri = {}
        positiveSec = {}
        negativeSec = {}
        positiveSecT = {}
        negativeSecT = {}

        fPri = open(fnamePri, 'wb')
        fSec = open(fnameSec, 'wb')  # Save model file as pickle
        pickle.dump(allPri, fPri)
        pickle.dump(allSec, fSec)
        fPri.close()
        fSec.close()

    if load == True:
        allPri = pickle.load(open(fnamePri, "rb"))
        allSec = pickle.load(open(fnameSec, "rb"))
        print("Train and Test data loaded")

    momentumlist = [0.2,0.3,0.4]
    lrlist= [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
    max_sent_list = [50,100,150]
    init_weight = 0.0001

    for momentum in momentumlist:
        for lr in lrlist:
            for max_sent in max_sent_list:
                classifierTrained=trainClassifier(allPri,allSec,lr,momentum,max_epoch, True,ftestRes,max_sent)
                correct,predicted_positives,predicted_negatives,true_positives,true_negatives,test_all_pos,test_all_neg = testClassifier(classifierTrained,allSec,max_sent)
                lrStr = ("%2.15f" % lr).rstrip('0')
                momentumStr = ("%2.15f" % momentum).rstrip('0')
                #prmStr = "%s-%s-%d-%d.bin" % (lrStr, momentumStr, max_epoch, batch_size)
                calculateAndPrintScores(ftestRes, 999,lrStr,momentumStr,max_sent,correct,predicted_positives,predicted_negatives,true_positives,true_negatives,
                                    test_all_pos,test_all_neg)
            #saveClassifierModel(classifierTrained, prmStr)

ftestRes.close()
print("End of classifier test")

