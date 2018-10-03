<h1>Experimentation with different word representations and classification methods</h1>

<h3>Dataset</h3>
The dataset is a database of newsgroup emails spanning a number of different topics. I used the dataset that split the data into training and testing sets in the proportion of 60:40. More details found here: http://qwone.com/~jason/20Newsgroups/.

<h3>Classification task</h3> 
Binary classification of documents into computer related and not computer related. 

<h3>Methods and algorithms used</h3>
3 text representation formats - Bag of words, TF-IDF and Word2Vec and 3 classification algorithms - Naive Bayes, Logistic Regression and Support Vector Machines.

<h3>Evaluation metric</h3>
F1 Score which is a good measure of both precision and recall.

<h3>Results</h3>
All statistics are related to validating the model on test data

<b>Bag of Words Representation</b>

Support Vector Machine Classification
F1 Score = 0.732076473712

Logistic Regression Classification
F1 Score = 0.737254381306

Naive Bayes Classification
F1 Score = 0.730217737653


<b>TF-IDF Representation</b>

Support Vector Machine Classification
F1 Score = 0.92843866171

Logistic Regression Classification
F1 Score = 0.901221455125

Naive Bayes Classification
F1 Score = 0.904275092937


<b>Word2Vec Representation</b>

Support Vector Machine Classification
F1 Score = 0.488449283059

Logistic Regression Classification
F1 Score = 0.74044078598
