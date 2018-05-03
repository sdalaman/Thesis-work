
# coding: utf-8


import os
import math
import pickle
import datetime
import numpy as np
import codecs

#import morfessor
import sys
import re
import os

ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
digits = '0123456789'
hexdigits = '0123456789abcdefABCDEF'
octdigits = '01234567'
printable = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
whitespace = ' \t\n\r\x0b\x0c'


class args(object):
    pass
    

def prepMorpDict(morphFile):
    dict = {}
    i = 0
    cnt = 0
    with codecs.open(morphFile, 'r','utf-8') as f:
        for line in f:
            i += 1
            if line[0] in punctuation or line[0] in digits:
                cnt += 1
                continue
            words = line.split(":-")
            w1 = words[0].strip()
            w2 = words[1].replace('\n','')
            w2 = w2.split('+')
            w2.remove('')
            w2s=''
            for i in range(len(w2)):
                w2s = w2s + w2[i] + ' \t '
            dict[w1]=w2s
    return dict

def prepMorphFile(infile,outfile,dict):
    out = codecs.open(outfile,'w','utf-8')
    with codecs.open(infile, 'r','utf-8') as f:
        for line in f:
            words = line.split()
            for i in range(len(words)):
                if words[i].strip() in dict.keys():
                    words[i] = dict[words[i]]
            newline = '\t'.join(words) + '\n'
            out.write(newline)
    out.close()

def convertDataMorph(dict,rpath,wpath):
    cnt = 0
    for file in os.listdir(rpath):
        fr = rpath + file
        fw = wpath + file[0:-4]+'.tok.morph'
        prepMorphFile(fr, fw, dict)
        cnt = cnt + 1
        print("file %s finished (fcnt=%d)" % (fr,cnt))

def convertFiles(nmIn,nmOut):
    rstr = ''
    os.chdir(cpath1)
    for lang in langs:
        if lang == 'en-tr':
            rstr = '_en'
            dict = dictEn
        if lang == 'tr-en':
            rstr = '_tr'
            dict = dictTr
        if lang == 'en-de':
            rstr = '_en'
            dict = dictEnDe
        if lang == 'de-en':
            rstr = '_de'
            dict = dictDe
        if lang == 'en-fr':
            rstr = '_en'
            dict = dictEnFr
        if lang == 'fr-en':
            rstr = '_fr'
            dict = dictFr
        cpath2w = cpath1 + lang + nmOut
        cpath2r = cpath1 + lang + nmIn + '/'
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
                    convertDataMorph(dict,cpath5r,cpath5w)

classes = ['art','arts','biology','business','creativity','culture','design','economics','education','entertainment','global','health','politics','science','technology']
#classes = ['art']
#langs = ['en-tr','tr-en']
#langs = ['de-en']
langs = ['en-fr','fr-en']
#langs = ['en-de','de-en','en-fr','fr-en']
types = ['train','test']
#types = ['test']
flags = ['positive','negative']
#flags = ['positive']
cpath1 = '/home/saban/work/python/pytorch-works/additive/data/ted/'


inMorphFileTr = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/turkish.all.tok.morph.list"
dictTr = prepMorpDict(inMorphFileTr)
inMorphFileEn = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/english.all.tok.morph.list"
dictEn = prepMorpDict(inMorphFileEn)
inMorphFileDe = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/deutch.all.tok.morph.list"
dictDe = prepMorpDict(inMorphFileDe)
inMorphFileFr = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/french.all.tok.morph.list"
dictFr = prepMorpDict(inMorphFileFr)
inMorphFileEnDe = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/train.de-en.en.tok.morph.list"
dictEnDe = prepMorpDict(inMorphFileEn)
inMorphFileEnFr = "/home/saban/work/python/works/polyglot/Morfessor-2.0.1/scripts/morfessor-models/train.en-fr.en.tok.morph.list"
dictEnFr = prepMorpDict(inMorphFileEn)


convertFiles('-tok','-morph/')

print("End ")

