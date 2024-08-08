# -*- coding: utf-8 -*-
# evaluate.py
# load into interpreter,
# then call find_best_combination(n), with n folds
# or single_features(n)
import nltk
import cPickle
from itertools import chain, combinations
from multiprocessing import Pool
import time

starttime = time.time()
extracted_features = cPickle.load(open('extracted_features.cpickle','r'))

# from the python generator cookbook
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
features = extracted_features[0][0].keys()
print "Number of features: " + str(len(features))
combinations = [x for x in powerset(features)]

def evaluate(data, k):
    f1_imp = 0
    f1_unimp = 0
    for j in range(k):
        traindata = [x for i,x in enumerate(data) if i % k == j]
        testdata = [x for i,x in enumerate(data) if i % k != j]
        cl = nltk.NaiveBayesClassifier.train(traindata)
        classified = cl.classify_many([feat for feat,tag in testdata])
        gold = [tag for feats,tag in testdata]
        matrix = nltk.ConfusionMatrix(gold, classified)
        if matrix[(0,0)] != 0:
            precision = float(matrix[(0,0)]) / (matrix[(0,0)] + matrix[(1,0)])
            recall = float(matrix[(0,0)]) / (matrix[(0,0)] + matrix[(0,1)])
#            if not ((recall == 0) | (precision == 0)):
            f1_imp += 2*precision*recall/(precision + recall)
        if matrix[(1,1)] != 0:
            precision1 = float(matrix[(1,1)]) / (matrix[(1,1)] + matrix[(0,1)])
            recall1 = float(matrix[(1,1)]) / (matrix[(1,1)] + matrix[(1,0)])
#            if not ((recall1 == 0) | (precision1 == 0)):
            f1_unimp += 2*precision1*recall1/(precision1 + recall1)
    return float(f1_imp) / k, float(f1_unimp) / k

def find_best_combination(folds=7):
    start = time.time()
    train_sets = []
    for combo in combinations:
        t_set = []
        for j in extracted_features:
            feats = dict()
            for i in combo:
                feats[i] = j[0][i]
            t_set.append((feats,j[1]))
        train_sets.append(t_set)
    print "Evaluating " + str(len(train_sets)) + " combinations using " + str(folds) + " folds..."
    m = 0
    for i,train_data in enumerate(train_sets):
        f1_imp, f1_unimp = evaluate(train_data, folds)
        if f1_imp > m: 
            print "Better combination found: "
            print "f1 imp.: " + str(f1_imp) + " f1 unimp.: " + str(f1_unimp) + " comb.: " + str(combinations[i])
            m = f1_imp
            mu = f1_unimp
            c = i
    print "Best F1-value resp. important sentences:   " + str(m)
    print "with F1-value resp. unimportant sentences: " + str(mu)
    print "achieved with combination: " + str(combinations[i])
    end = time.time()
    print 'Elapsed time: ' + str(round(end - start ,2)) + ' seconds'
    print 'Average time per feature: ' + str(round((end - start)/len(train_sents)) ,2) + ' seconds'
    
def find_best_combination2(folds=7):
    start = time.time()
    train_sets = []
    for combo in combinations:
        t_set = []
        for j in extracted_features:
            feats = dict()
            for i in combo:
                feats[i] = j[0][i]
            t_set.append((feats,j[1]))
        train_sets.append(t_set)
    print "Evaluating " + str(len(train_sets)) + " combinations using " + str(folds) + " folds..."
    
    p = Pool(1)
    
    results = [p.apply(evaluate, args=(train_data,folds)) for train_data in train_sets]
    
    i = 0
    bestm = 0
    bestcombi = 0
    for m,mu in results:
        
        if m >= bestm:
            bestm = m
            bestcombi = i
        
        #print "Best F1-value resp. important sentences:   " + str(m)
        #print "with F1-value resp. unimportant sentences: " + str(mu)
        #print "achieved with combination: " + str(combinations[i])
        
        i+=1
    
    print "done"
    end = time.time()
    print 'Elapsed time: ' + str(round(end - start ,2)) + ' seconds'
    print 'Average time per feature: ' + str(round((end - start)/len(train_sets) ,2)) + ' seconds'
    print '___________________________________'
    print "Best combination found: "
    print "f1 imp.: " + str(results[bestcombi][0]) + " f1 unimp.: " + str(results[bestcombi][0]) + " comb.: " + str(combinations[bestcombi])
    
    
    
    
    
    
    
    
def single_features(folds=7):
    start = time.time()
    train_sets = []
    for i in features:
        t_set = []
        for j in extracted_features:
            feats = dict()
            feats[i] = j[0][i]
            t_set.append((feats,j[1]))
        train_sets.append(t_set)
    for i,train_data in enumerate(train_sets):
        f1_imp, f1_unimp = evaluate(train_data, folds)
        print "f1 imp.: " + str(f1_imp) + " f1 unimp.: " + str(f1_unimp) + " comb.: " + str(combinations[1+i])
#features = extracted_features[0][0].keys()
##combinations = features
#combinations = [x for x in features]
#train_sets = []
#for combo in combinations:
#    t_set = []
#    for j in extracted_features:
#        feats = dict()
#        for i in combo:
#            feats[i] = j[0][i]
#        t_set.append((feats,j[1]))
#    train_sets.append(t_set)
endtime = time.time()
print 'Elapsed time: ' + str(round(endtime - starttime ,2)) + ' seconds'