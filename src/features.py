# -*- coding: utf-8 -*-
# features.py
# here we define the functions for our features
import datetime
import nltk
import math
import cPickle
import re

from prepare_corpus import normalize, frequency_complete, frequency_thread

from nltk.corpus import wordnet as wn

_, freq, freq_list, freq_imp, _ = cPickle.load(open('prepared_corpus.cpickle','rb'))

# contains >, url(\.(3)com\), wordnet, pos % 5, contains x types from all important sentences
# reminder, remember, forget, 
def sent_contains(sent,words):
    for i in words:
        if sent.find(i)>= 0:
            return 1
    return 0

# Berechnet die IDF (ueber alle Threads) eines Wortes
def idf(word):
    counter = 0
    for freq in freq_list:
        if word in freq:
            counter += 1
    return len(freq_list)/float(counter)

# Berechnet die TF (ueber alle Threads) eines Wortes    
def tf(word):
    return float(freq[word])/max(freq.values())  

# Berechnet die TF-IDF (Produkt aus TF und IDF s.o.) fuer jedes Wort im Satz und gibt die Summe zurück.
def tfidf(sent):
    sent_tokens = nltk.word_tokenize(sent)
    n_sent = normalize(sent_tokens)
    sent_sum = 0
    for n_word in n_sent:
        sent_sum += tf(n_word)*math.log(idf(n_word))
    return round(float(sent_sum)/len(n_sent),1)

# Zaehlt die Woerter im Satz
def f_wordcount(sent):
    return len(nltk.word_tokenize(sent))

# Zaehlt die Woerter im Satz und teilt das Ergebnis durch 10 (gibt den Integer-Wert zurück)
def f_wordcount10(sent):
    return len(nltk.word_tokenize(sent)) / 10

# Berechnet die Relevanz eines Satzes zur Betreffzeile (hier simpel durch Zaehlen gleicher Wortvorkommen)
def f_topic_relevance(sent, subject):
    senttokens = nltk.word_tokenize(sent)
    subjecttokens = nltk.word_tokenize(subject)
    counter = 0
    norm_senttokens = normalize(senttokens)
    norm_subjecttokens = normalize(subjecttokens)
    for token in norm_subjecttokens:
        for compare in norm_senttokens:
            if token  == compare:
                counter += 1
    return round(float(counter)/len(norm_subjecttokens),1)

# Prueft den Satz auf ein Synonym von "important" (signalisiert Wichtigkeit)
def f_contains_importance(sent):
    importance = ["important", "big","critical","crucial","decisive","essential","extensive","far-reaching","great","imperative","influential","large","meaningful","necessary","paramount","relevant","serious","significant","urgent","vital","big-league","chief","considerable","conspicuous","determining","earnest","esteemed","exceptional","exigent","foremost","front-page","grave","heavy","importunate","marked","material","matter","momentous" "of moment","of note","of substance","ponderous","pressing","primary","principal","salient","signal","something","standout","weighty"]
    return sent_contains(sent, importance)
# def f_wn_important(sent):
#    p = nltk.PorterStemmer()
#    syns = list(set([p.stem(x) for c in wn.synsets('important') for x in c.lemma_names()]))
#    return sent_contains(sent,syns)

# Prueft den Satz darauf, ob "business"-Sprache benutzt wurde. (wichtig)
def f_business_language(sent):
    words = ['fax', 'asap', 'mail', 'call', 'office', 'meeting', 'week', 'word', 'done', 'day', 'need', 'people', 'group', 'finish', 'complete', 'think', 'possible', 'problem']
    return sent_contains(sent,words) 

# Prueft den Satz auf Begruessungs- und Abschiedsformeln (unwichtig)
def f_greeting(sent):
    words = ["hello","hi","hey","bye","good to meet", "nice to meet", "see you", "cheers", "regards", "sincerely", "yours truly", "yours faithfully"]
    return sent_contains(sent,words) 
# def f_greetings(sent):
#     p = nltk.PorterStemmer()
#     special = "regards cheers sup".split()
#     for i in "hi hello bye".split():
#         special.append([p.stem(x) for c in wn.synsets(i) for x in c.lemma_names()])

# Berechnet die relative Position des Satzes innerhalb der Mail
def f_sent_rel_pos(sent, sents):
    s = [sentence for tag,sentence in sents]
    return round( (s.index(sent)+1) / float(len(s)) , 1)

# Prueft den Satz auf Verben, die einen Plan ausdruecken
def f_contains_plan(sent):
    taskword_list = ["must","need","have to", "has to", "going to", "will", "'ll", "want", "would", "could"]
    return sent_contains(sent, taskword_list)

# Sucht im Satz nach Fragezeichen (Hinweis auf Frage)
def f_contains_question(sent):
    if sent.find("?")>= 0:
        return 1
    else:
        return 0
    
# Sucht im Satz nach Ausrufezeichen (Hinweis auf Ausruf)
def f_contains_exclamation(sent):
    if sent.find("!") >= 0:
        return 1
    else:
        return 0

# In diesem Corpus sind Zitate mit "&gt;" und ">" markiert. Nach diesen Ausdruecken wird am Satzanfang gesucht.   
def f_sent_is_quote(sent):
    if (sent[0:4] == "&gt;" or sent[0] == ">"):
        return 1
    else:
        return 0

# Prueft den Satz auf eine Floskel, die eine eigene Meinung einleitet.
def f_contains_opinion(sent):
    opinion_phrases = ["i think", "in my opinion", "i believe", "from my point of view"]
    sent_contains(sent, opinion_phrases)

# Berechnet fuer jedes Wort im Satz einen "Seltenheitsfaktor" (im Kontext des jeweiligen Threads) und gibt die Summe fuer den kompletten Satz (normalisiert durch die Satzlaenge) zurueck.
def specific(sent, threadnumber):
    sent_tokens = nltk.word_tokenize(sent)
    n_sent = normalize(sent_tokens)
    spec_sum = 0
    thread_freq = freq_list[threadnumber]
    for word in n_sent:
        spec_sum += thread_freq[word]/float(freq[word])
    return round(float(spec_sum)/len(n_sent),1)

# Sucht im Satz nach Zeitangaben 
def f_contains_time(sent):
    time_list = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "o'clock"]
    for time in time_list:
        if sent.find(time) >= 0:
            return 1
    if re.search("\dam", sent):
        return 1
    if re.search("\dpm", sent):
        return 1
    return 0

# Konvertiert eine Zeitangabe in einem String zu einem Datetime-Objekt (unter Einbezug der Zeitverschiebung)
def get_utc_time(time):
    date_p1 = time[:20] + time[-4:]
    delta_hour = time[21:23]
    delta_minute = time[23:25]
    delta = time[20]
    strp = datetime.datetime.strptime(date_p1, '%a %b %d %H:%M:%S %Y')
    strp2 = datetime.timedelta(hours=int(delta_hour), minutes=int(delta_minute))
    if (delta == "+"):
        strp += strp2
    else:
        strp -= strp2
    return strp

# Berechnet unter Angabe zweier Strings (die Datum und Uhrzeit beinhalten) ein Timedelta-Objekt, das den zeitlichen Unterschied angibt    
def get_time_difference(time1, time2):
    datetime1 = get_utc_time(time1)
    datetime2 = get_utc_time(time2)
    if (datetime1>datetime2):
        return datetime1 - datetime2
    else:
        return datetime2 - datetime1

# Berechnet unter Angabe zweier Strings den zeitlichen Unterschied in Sekunden    
def get_time_difference_in_seconds(time1, time2):
    diff = get_time_difference(time1, time2)
    return diff.days * 24 * 3600 + diff.seconds

# Berechnet die relative Position der Mail in der Zeitspanne zwischen Anfang und Ende des Threads.
def f_rel_time(start, time, end):
    thread = get_time_difference_in_seconds(start, end)
    mail = get_time_difference_in_seconds(start, time)
    return round(float(mail)/thread,1)
