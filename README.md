# Extractive Textual Summarization

Overview:

The goal: Selecting important 
sentences in Email conversations
I use the Naive Bayes Machine 
learning algorithm  
Potential use cases:  
  • automatic summarizing  
  • highlighting sentences for faster 
    overview in a user interface  

_______________________________


Data ressources:

BC3 Email Corpus: 
40 Email threads (3222 sentences)  
wordnet:  
a lexical ressource we used for compiling wordlists

_______________________________


Implementation:

prepare_corpus.py  
    -convert corpus into python datastructure  
features.py  
    -collection of feature functions  
extract_features.py  
    -create full featureset for training  
evaluate.py  
    -split featureset and compare performance of parts  

Intermediate results are stored in cPickle files

_______________________________


Experiments & Results:

Comparing F1-scores after training the classifier
with different combinations of features.  
Multiple tests on same data for more reliable
results (k-fold evaluation)

![features_chart](/img/stats.jpg?raw=true "Features")

_______________________________


Conclusion:

Combination of features is crucial.
Single features may have bad performance.  
Testing all possible feature combinations is
difficult, as the time needed for n features is
2^n in the best case.  
Splitting feature extraction and evaluation
speeds this up.



