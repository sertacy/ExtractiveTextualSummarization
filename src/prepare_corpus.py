# -*- coding: utf-8 -*-
# prepare_corpus.py
# parses the mail corpus and the annotation corpus,
# combines the data into a python datastructure,
# and computes word frequencys for further use
import xml.etree.ElementTree as ET
import nltk
import random
import math
import cPickle
import time

starttime = time.time()

corpus = ET.parse("bc3corpus.1.0/corpus.xml")
annotation = ET.parse("bc3corpus.1.0/annotation.xml")
corpus_root = corpus.getroot()
annotation_root = annotation.getroot()

def gettext(elem):
    text = elem.text or ","
    for e in elem:
        text += gettext(e)
        if e.tail:
            text += e.tail
    return text

anno = dict()

for thread in annotation_root:
    imp = set()
    for annotation in thread[2:]:
        for sent in annotation[3]:
           imp |= set([sent.attrib['id']])
    anno[gettext(thread[0])] = sorted(list(imp))

data = []
pos = 0
neg = 0

for thread in corpus_root:
    t = dict()
    t['name'] = gettext(thread[0])
    t['listno'] = gettext(thread[1])
    mails = []
    for doc in thread[2:]:
        d = dict()
        d['received'] = gettext(doc[0])
        d['from'] = gettext(doc[1])
        d['to'] = gettext(doc[2])
        d['subject'] = gettext(doc[3])
        s = []
        for j in doc[4]:
            sent = gettext(j).strip(' \n')
            if j.attrib['id'] in anno[t['listno']]:
                s.append((1,sent))
                pos += 1
            else:
                s.append((0,sent))
                neg += 1
        d['sents'] = s
        mails.append(d)
    t['mails'] = mails
    data.append(t)

def normalize(wordlist):
    lower = [x.lower() for x in wordlist]
    # ! to be tested
    for i in lower:
        for n in i:
            if n not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                i.replace(n,'')
    porter = nltk.PorterStemmer()
    stemmed = [porter.stem(t) for t in lower]
    wnl = nltk.WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(t) for t in lower]
    return [x for x in lemmatized if x not in nltk.corpus.stopwords.words("english")]

def frequency_complete(data):
    frequency = {}
    for thread in data:
        for mail in thread['mails']:
            for tag,sent in mail['sents']:
                tokens = nltk.word_tokenize(sent)
                n_tokens = normalize(tokens)
                for t in n_tokens:
                    if t not in frequency:
                        frequency[t] = 1
                    else:
                        frequency[t] += 1
    return frequency

def frequency_thread(thread):
    frequency = {}
    for mail in thread['mails']:
        for tag,sent in mail['sents']:
            tokens = nltk.word_tokenize(sent)
            n_tokens = normalize(tokens)
            for t in n_tokens:
                if t not in frequency:
                    frequency[t] = 1
                else:
                    frequency[t] += 1
    return frequency

def max_wordcount_thread(thread):
    max_wordcount = 0
    for mail in thread['mails']:
        for tag,sent in mail['sents']:
            wordcount = len(nltk.word_tokenize(sent))
            if (wordcount>max_wordcount):
                max_wordcount = wordcount
    return max_wordcount

def max_wordcount_mail(mail):
    max_wordcount = 0
    for tag,sent in mail['sents']:
        wordcount = len(nltk.word_tokenize(sent))
        if (wordcount>max_wordcount):
            max_wordcount = wordcount
    return max_wordcount
##
##
##
##
##
##
## legal / good to compute the frequencys before splitting into test and training ??
##
##
##
##
##
def frequency_in_important_s(data):
    frequency_i = {}
    frequency_u = {}
    for thread in data:
        for mail in thread['mails']:
            for tag,sent in mail['sents']:
                tokens = nltk.word_tokenize(sent)
                n_tokens = normalize(tokens)
                if tag == 1:
                    for t in n_tokens:
                        if t not in frequency_i:
                            frequency_i[t] = 1
                        else:
                            frequency_i[t] += 1
                if tag == 0:
                    for t in n_tokens:
                        if t not in frequency_u:
                            frequency_u[t] = 1
                        else:
                            frequency_u[t] += 1
    i = sorted(frequency_i.items(), key=lambda x: x[1])
    u = sorted(frequency_u.items(), key=lambda x: x[1])
    return i[-50:],u[-50:]

freq_imp = frequency_in_important_s(data)
freq = frequency_complete(data)
freq_list = []
max_list = []

for thread in data:
    freq_list.append(frequency_thread(thread))
    max_list.append(max_wordcount_thread(thread))

print '# E-mails: ' + str(sum( [len(t['mails']) for t in data] ))
print '# E-mail threads: ' + str(len(data))
print '# Important sentences: ' + str(pos)
print '# Unimportant sentences: ' + str(neg)
print '% of sentences that are important: ' + str(pos/(pos+float(neg)))
print '# of Types in corpus: ' + str(len(freq))

cPickle.dump((data, freq, freq_list, freq_imp, max_list), open('prepared_corpus.cpickle', 'wb'), protocol=-1)
endtime = time.time()

print 'Elapsed time: ' + str(round(endtime - starttime ,2)) + ' seconds'
