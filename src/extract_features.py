# -*- coding: utf-8 -*-
# extract_features.py
import nltk
import random
import math
import cPickle
from features import *
import time
starttime = time.time()

data, _, _, _, max_list = cPickle.load(open('prepared_corpus.cpickle','rb'))

def max_wordcount_mail(mail):
    max_wordcount = 0
    for tag,sent in mail['sents']:
        wordcount = len(nltk.word_tokenize(sent))
        if (wordcount>max_wordcount):
            max_wordcount = wordcount
    return max_wordcount

train_data = []
threadnumber = -1

for thread in data:
    threadnumber += 1
    mailnumber = 0
    op = thread['mails'][0]['from']
    start = thread['mails'][0]['received']
    end = thread['mails'][len(thread['mails']) - 1]['received']
    for mail in thread['mails']:
        mailnumber += 1
        for tag,sent in mail['sents']:
            train_data.append(({#'sentence length' : len(sent),
                                #'?' : f_contains_question(sent),
                                #'!' : f_contains_exclamation(sent),
                                'relevance to topic' : f_topic_relevance(sent, thread['name']),
                                '# recepients' : len(mail['to'].split(",")),
                                'rel. position of sent.' : f_sent_rel_pos(sent,mail['sents']),
                                'rel. position of mail' : round(float(mailnumber) / len(thread['mails']),1),
                                'rel. time of mail' : f_rel_time(start, mail['received'], end),
                                ####'tfidf of sentence' : tfidf(sent),
                                #'sent_abs_pos_o' : [s for _,s in mail['sents']].index(sent),
                                ####'specific of sentence' : specific(sent, threadnumber),
                                #'wordcount' : f_wordcount(sent),
                                #'wordcount/10' : f_wordcount10(sent),
                                ####'norm. wordc. global': int(f_wordcount(sent) * (float(f_wordcount(sent))/max_list[threadnumber]))/10,
                                ####'norm. wordc. local': int(f_wordcount(sent) * (float(f_wordcount(sent))/max_wordcount_mail(mail)))/10,
                                ####'important' : f_contains_importance(sent.lower()),
                                ###'quote': f_sent_is_quote(sent),
                                ###'opinion': f_contains_opinion(sent.lower()),
                                #'dummy feature' : '42',
                                ###'from_op' : mail['from'] == op,
                                ###'to_op' : mail['to'] == op,
                                ###'time' : f_contains_time(sent.lower()),
                                ###'business' : f_business_language(sent.lower()),
                                ###'greeting' : f_greeting(sent.lower()),
                                'plan': f_contains_plan(sent.lower())
                                },
                                tag)) 
#Fixed seed for deterministic shuffling
#So when we modify some feature, and run the evaluation again,
#the change in the result is not because of a different shuffling
def seed():
#    return 0.39914190079341594
    return 0.48388069099035114
random.shuffle(train_data, seed)
cPickle.dump(train_data, open('extracted_features.cpickle', 'wb'), protocol=-1)
endtime = time.time()
print 'Elapsed time: ' + str(round(endtime - starttime ,2)) + ' seconds'
