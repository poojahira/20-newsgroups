from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover,HashingTF,IDF,CountVectorizer,Word2Vec 
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.classification import *
from pyspark.ml.classification import *
from pyspark.sql.functions import *
from pyspark.ml.evaluation import *
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import re

if __name__ == "__main__":

    	sc = SparkContext(appName="20 newsgroups classification")
	sqlContext = SQLContext(sc)
	# get data from training directory and clean it up
	rdd = sc.wholeTextFiles("data/20news-bydate-train/*/*").map(lambda x:(x[0].split('/')[7],x[1],x[0].split('/')[6]))
	rdd1 = rdd.filter(lambda x:x[2] == 'alt.atheism').map(lambda x:(x[0],x[1],0.0))
	rdd2 = rdd.filter(lambda x:'religion.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd3 = rdd.filter(lambda x:'rec.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd4 = rdd.filter(lambda x:'comp.' in x[2]).map(lambda x:(x[0],x[1],1.0))
	rdd5 = rdd.filter(lambda x:'sci.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd6 = rdd.filter(lambda x:'politics.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd7 = rdd.filter(lambda x:'forsale' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd = sc.union([rdd1,rdd2,rdd3,rdd4,rdd5,rdd6,rdd7]).cache()
	rdd = rdd.map(lambda x:(x[0],re.sub('-----BEGIN PGP SIGNED MESSAGE-----','',x[1]),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('^From:.*', '', x[1],flags=re.MULTILINE),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('^NNTP-Posting-Host:.*', '', x[1],flags=re.MULTILINE),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('^Nntp-Posting-Host:.*', '', x[1],flags=re.MULTILINE),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('^.*writes:.*', '', x[1],flags=re.MULTILINE),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('^Organization:.*', '', x[1],flags=re.MULTILINE),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('[^-,.\nA-Za-z ]+','', x[1]),x[2]))
	rdd = rdd.map(lambda x:(x[0],re.sub('[^-A-Za-z ]+',' ',x[1]),x[2]))
	myre = re.compile("^--.*")
	rdd = rdd.map(lambda x:(x[0],myre.split(x[1])[0],x[2]))
	rdd = rdd.map(lambda x:(int(x[0]),x[1].lower(),x[2]))
	df = rdd.toDF(['ID','Content','label'])

	#get data from testing directory and clean it up
	test_rdd = sc.wholeTextFiles("data/20news-bydate-test/*/*").map(lambda x:(x[0].split('/')[7],x[1],x[0].split('/')[6]))
	rdd1 = test_rdd.filter(lambda x:x[2] == 'alt.atheism').map(lambda x:(x[0],x[1],0.0))
	rdd2 = test_rdd.filter(lambda x:'religion.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd3 = test_rdd.filter(lambda x:'rec.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd4 = test_rdd.filter(lambda x:'comp.' in x[2]).map(lambda x:(x[0],x[1],1.0))
	rdd5 = test_rdd.filter(lambda x:'sci.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd6 = test_rdd.filter(lambda x:'politics.' in x[2]).map(lambda x:(x[0],x[1],0.0))
	rdd7 = test_rdd.filter(lambda x:'forsale' in x[2]).map(lambda x:(x[0],x[1],0.0))
	test_rdd = sc.union([rdd1,rdd2,rdd3,rdd4,rdd5,rdd6,rdd7]).cache()
	test_rdd = test_rdd.map(lambda x:(x[0],re.sub('-----BEGIN PGP SIGNED MESSAGE-----','',x[1]),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('^From:.*', '', x[1],flags=re.MULTILINE),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('^NNTP-Posting-Host:.*', '', x[1],flags=re.MULTILINE),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('^Nntp-Posting-Host:.*', '', x[1],flags=re.MULTILINE),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('^.*writes:.*', '', x[1],flags=re.MULTILINE),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('^Organization:.*', '', x[1],flags=re.MULTILINE),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('[^-,.\nA-Za-z ]+','', x[1]),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],re.sub('[^-A-Za-z ]+',' ',x[1]),x[2]))
        test_rdd = test_rdd.map(lambda x:(x[0],myre.split(x[1])[0],x[2]))
	test_rdd = test_rdd.map(lambda x:(int(x[0]),x[1].lower(),x[2]))
	test_df = test_rdd.toDF(['ID','Content','label'])
	
	# set up ML pipelines
	regexTokenizer = RegexTokenizer(inputCol="Content", outputCol="words",pattern="\\W")
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	word2Vec = Word2Vec(vectorSize=300, minCount=5, inputCol="filtered", outputCol="features")
	countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000)

	pipeline = []
	rep = []
	pipeline.append(Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors]))
	rep.append("Bag of Words")
	pipeline.append(Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF,idf]))
	rep.append("TF-IDF")
	pipeline.append(Pipeline(stages=[regexTokenizer, stopwordsRemover, word2Vec]))	
	rep.append("Word2Vec")
	
	f = open('output.txt', 'w')
	f.write("All statistics are related to validating the model on test data\n")

	# run 3 algorithms on all data representations
	for i in range(0,3):
		pipelineFit = pipeline[i].fit(df)
		dataset = pipelineFit.transform(df).cache()
		pipelineFit_test = pipeline[i].fit(test_df)
                dataset_test = pipelineFit_test.transform(test_df).cache()
		f.write("\n\n%s Representation\n" % rep[i])
		
		#SVM
		n = (dataset.select(col("label"), col("features")).rdd.map(lambda row: LabeledPoint(row.label, row.features)))
		n_test = (dataset_test.select(col("label"), col("features")).rdd.map(lambda row: LabeledPoint(row.label, row.features)))
		model_svm = SVMWithSGD.train(n, iterations=100)
		labelsAndPreds = n_test.map(lambda p: (p.label, model_svm.predict(p.features)))
		predsandlabels = labelsAndPreds.map(lambda x:(float(x[1]),float(x[0])))
		f.write("\nSupport Vector Machine Classification\n")
                metrics_svm1 = MulticlassMetrics(predsandlabels)
                f.write("F1 Score = %s\n" % metrics_svm1.fMeasure())

		#Logistic Regression
		lr = LogisticRegression(maxIter=100, regParam=0.1)
		model_lr = lr.fit(dataset)
		pred_test = model_lr.transform(dataset_test)
		predictionsAndLabels = pred_test.select(pred_test.prediction,pred_test.label).rdd
		f.write("\nLogistic Regression Classification\n")
                metrics_reg1 = MulticlassMetrics(predictionsAndLabels)
                f.write("F1 Score = %s\n" % metrics_reg1.fMeasure())

		#Naive Bayes
		if i==2:
			continue
                nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
                model_nb = nb.fit(dataset)
                f.write("\nNaive Bayes Classification\n")
                pred_test = model_nb.transform(dataset_test)
                predictionsAndLabels = pred_test.select(pred_test.prediction,pred_test.label).rdd
		metrics_nb1 = MulticlassMetrics(predictionsAndLabels) 
                f.write("F1 Score = %s\n" % metrics_nb1.fMeasure())
	
	f.close()	
	sc.stop()
