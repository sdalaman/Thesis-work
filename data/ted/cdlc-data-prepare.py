
# coding: utf-8


import os
import math
import pickle
import datetime
import numpy as np
import nltk
import codecs


class args(object):
    pass
    
def convertData(rpath,wpath,rs):
    #print("read path : %s" % rpath)
    rfcnt = wfcnt = 0
    rlen = wlen = 0
    for file in os.listdir(rpath):
        #print("file : %s" % file)
        fw = open(wpath + "/" + file[0:-4], 'w')
        with open(rpath + "/" + file, 'r') as fr:
            rfcnt += 1
            text = fr.read()
            rlen += len(text)
            text=text.replace(rs, "")
            wlen += len(text)
            fw.write(text)
            wfcnt += 1
            #    words = text.split(" ")
    return rfcnt,wfcnt,rlen,wlen,0,0

def convertDataTok(rpath,wpath,rs):
    rfcnt = wfcnt = 0
    rlen = wlen = 0
    rsen = wsen = 0
    max_len = 1000
    for file in os.listdir(rpath):
        fr = codecs.open(rpath + "/" + file, 'r', 'utf-8')
        fw = codecs.open(wpath + "/" + file[0:-4]+'.tok', 'w', 'utf-8')
        rfcnt += 1
        wfcnt += 1
        data = fr.read()
        rlen += len(data)
        data = data.replace(rs, "")
        wlen += len(data)
        sents = data.split('\n')
        no_of_sent = len(sents)
        rsen += no_of_sent
        for i in range(no_of_sent):
            tokens = nltk.word_tokenize(sents[i])
            if len(tokens) < max_len :
                fw.write('\t'.join(tokens) + '\n')
                wsen += 1
    return rfcnt,wfcnt,rlen,wlen,rsen,wsen

classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology']
#classes = ['art']
#langs = ['en-tr','tr-en']
langs = ['en-de','de-en','en-fr','fr-en']
types = ['train','test']
flags = ['positive','negative']
cpath1 = '/home/saban/work/python/pytorch-works/additive/data/ted/'

def convertFiles(nm):
    rstr = ''
    rtot =  wtot = 0
    rltot = wltot = 0
    rstot = wstot = 0
    rc = wc = rs = ws = rl = wl = 0
    os.chdir(cpath1)
    for lang in langs:
        if lang == 'en-de':
            rstr = '_en'
        if lang == 'de-en':
            rstr = '_de'
        if lang == 'en-fr':
            rstr = '_en'
        if lang == 'fr-en':
            rstr = '_fr'
        cpath2w = cpath1 + lang + nm
        cpath2r = cpath1 + lang + '/'
        if not os.path.exists(cpath2w):
            os.mkdir(cpath2w)
        for type in types:
            cpath3w = cpath2w + type + '/'
            cpath3r = cpath2r + type + '/'
            if not os.path.exists(cpath3w):
                os.mkdir(cpath3w)
            for classname in classes:
                cpath4w = cpath3w + classname + '/'
                cpath4r = cpath3r + classname + '/'
                if not os.path.exists(cpath4w):
                    os.mkdir(cpath4w)
                for flag in flags:
                    cpath5w = cpath4w + flag + '/'
                    cpath5r = cpath4r + flag + '/'
                    if not os.path.exists(cpath5w):
                        os.mkdir(cpath5w)
                #print(cpath5)
                    if nm == '-tok/':
                        rc, wc, rl, wl, rs, ws = convertDataTok(cpath5r,cpath5w,rstr)
                    elif nm == '-cl/':
                        rc, wc, rl, wl, rs, ws = convertData(cpath5r, cpath5w, rstr)
                    rtot += rc
                    wtot += wc
                    rltot += rl
                    wltot += wl
                    rstot += rs
                    wstot += ws
                    if nm == '-cl/':
                        print("%s : %d -- %s : %d\n" % (cpath5r,rc,cpath5w,wc))
                    elif nm == '-tok/':
                        print("%s : %d -- %s : %d\n" % (cpath5r, rs, cpath5w, ws))

    print("read file cnt %d -- write file cnt %d\n" % (rtot,wtot))
    print("read len %d -- write len %d\n" % (rltot,wltot))
    print("read file sentence cnt %d -- write file sentence cnt %d\n" % (rstot, wstot))


convertFiles('-cl/')
convertFiles('-tok/')

print("End ")

