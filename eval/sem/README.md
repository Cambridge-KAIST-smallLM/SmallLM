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
