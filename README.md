# Extractive Email Conversations Summarization

## Overview

This project implements extractive summarization for email conversations using a Naive Bayes  
machine learning algorithm. The system selects the most important sentences from email threads  
to create concise summaries.

### Potential Use Cases
- Automatic email summarization
- Intelligent sentence highlighting for faster content scanning
- Enhanced user interface for email clients

## Data Resources

- **BC3 Email Corpus**: 40 email threads containing 3,222 sentences
- **WordNet**: A lexical database used for compiling feature word lists

## Implementation

The project consists of four main modules:

- `prepare_corpus.py` - Converts the corpus into a Python data structure
- `features.py` - Collection of feature extraction functions
- `extract_features.py` - Generates complete feature sets for model training
- `evaluate.py` - Splits feature sets and evaluates classifier performance

Intermediate results are stored as cPickle files for efficient processing.

## Experiments & Results

The classifier's performance was evaluated using F1-scores across different feature combinations.   
K-fold cross-validation was employed to ensure reliable and reproducible results.

![Performance Statistics](/img/stats.jpg?raw=true)

## Conclusion

Key findings from this research:

- **Feature combination is critical** - Individual features often perform poorly in isolation,
  while combinations yield significantly better results
- **Computational complexity** - Testing all possible feature combinations for *n* features
  requires O(2^n) evaluations in the best case, making exhaustive search impractical
- **Optimization strategy** - Separating feature extraction from evaluation substantially
  reduces computational time and enables more efficient experimentation

## Future Work

Potential areas for improvement include exploring more sophisticated feature selection algorithms and testing with larger email corpora.
