# Semantic Evaluation 

Semantic Similarity is defined in various standard evaluation metrics: 

- SimLex999: nouns, verbs, adjectives (mainly for nouns and adjectives) [0,10] Scale I think??. See https://fh295.github.io/simlex.html. 
- SimVerb3500 (lots of verbs) [0, 10] scale. 
- Card660 [0, 10]

Some other datasets (historic value) are also included. 

These define word pairs grouped based on semantic similarity with a gold correlation score based on human annotators. 

These tests in combination provide a holistic measure of model preference for semantic and conditional similarity. 

The most important evaluation is Card600 (rare word similarity). 

Conditional Similarity (frequency contingent):   

.....................................................
SimVerb-3500-stats.txt – most important is SIMRNKV [within the dataset], but there are also some statistics for more general reference in two columns  [BNCFREQ and BNCRNKV]

This file contains additional statistics per verb lemma regarding frequency (extracted from BNC) and VerbNet class membership.

COUNTER: Numbers the lemma in the file, goes from 1-827; all verb lemmas were sorted alphabetically.
VBLEMMA: Verb lemma.
VBCLASS: The list of classes (delimited by ;) to which the particular verb lemma belongs (can be more than 1, N/A if they don't belong to any class). There are 101 Levin-style classes as in VerbNet.
BNCFREQ: Absolute frequency in the BNC corpus.
BNCRNKW: Absolute ranking of verb lemma given all lemmas and their frequencies from the BNC corpus.
BNCRNKV: Absolute ranking of verb lemma given only verb lemmas and their frequencies from the BNC corpus.
SIMRNKV: Relative ranking of verb lemma using only the set of 827 verb lemmas from our pool of verb types (values: 1-827).
BNCQRTV: The quartile to which the verb lemma belongs given the set of all verb lemmas and their frequencies from the BNC corpus.
