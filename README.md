Dataset: The dataset is a database of newsgroup emails spanning a number of different topics. I used the dataset that split the data into training and testing sets in the proportion of 60:40 found here: http://qwone.com/~jason/20Newsgroups/.

Classification task: Binary classification of documents into computer related and not computer related. 

Methods and algorithms used: 3 text representation formats - Bag of words, TF-IDF and Word2Vec and 3 classification algorithms - Naive Bayes, Logistic Regression and Support Vector Machines.

Evaluation metric: F1 Score which is a good measure of both precision and recall.

Output:

All statistics are related to validating the model on test data

Bag of Words Representation

Support Vector Machine Classification
F1 Score = 0.732076473712

Logistic Regression Classification
F1 Score = 0.737254381306

Naive Bayes Classification
F1 Score = 0.730217737653


TF-IDF Representation

Support Vector Machine Classification
F1 Score = 0.92843866171

Logistic Regression Classification
F1 Score = 0.901221455125

Naive Bayes Classification
F1 Score = 0.904275092937


Word2Vec Representation

Support Vector Machine Classification
F1 Score = 0.488449283059

Logistic Regression Classification
F1 Score = 0.74044078598
