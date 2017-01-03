# -*- encoding:UTF-8 -*-
#-- Author: TNT_000 by Abner yang
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
import gensim
import Cython
import numpy as np 
import pandas as pd 
import jieba
import jieba.posseg
import jieba.analyse
import time
import re
import codecs
import sys 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import IncrementalPCA
reload(sys) 
sys.setdefaultencoding('utf8') 

#-- Log function
def LogInfo(stri):
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' '+stri
#-- vote blend
def getVote(result):
	predict = []
	for line in result:
		maxVal = -1
		maxNum = -1
		dictNum = defaultdict(int)
		for values in line:
			dictNum[values] += 1
			if maxNum < dictNum[values]:
				maxVal = values
				maxNum = dictNum[values]
		predict.append(maxVal)
	return predict
#-- get word feom docments
def getWord():
	stopwords = {}
	for line in codecs.open('../data/stop.txt', 'r', 'utf-8'):
		stopwords[line.rstrip()] = 1

	data = pd.read_csv('../data/train.csv', header = 0, sep = ',', encoding = 'utf-8')
	data1 = data['qlist'].values.T

	text = []
	lineNum = 0
	for line in data1:
		lineNum += 1
		listq = line.split(' ')
		wordList = ''
		num = 0
		for query in listq:
			results=re.findall("(?isu)(https?\://[a-zA-Z0-9\.\?/&\=\:]+)", query)
			if len(results) == 0:
				reg_list = jieba.cut(query)
				for reg in reg_list:
					#print reg
					if reg.rstrip() not in stopwords:
						if num == 0:
							wordList = reg
							num = 1
						else:
							wordList = wordList + ' ' + reg
		text.append(wordList)
		if lineNum%1000 == 0:
			print lineNum

	d = pd.DataFrame({'uid':data['uid'].values.T, 'qlist':text})
	d.to_csv('../feature/trainQlist.csv', index = False)

	data = pd.read_csv('../data/test.csv', header = 0, sep = ',', encoding = 'utf-8')
	data1 = data['qlist'].values.T

	text = []
	lineNum = 0
	for line in data1:
		lineNum += 1
		listq = line.split(' ')
		wordList = ''
		num = 0
		for query in listq:
			results=re.findall("(?isu)(https?\://[a-zA-Z0-9\.\?/&\=\:]+)", query)
			if len(results) == 0:
				reg_list = jieba.cut(query)
				for reg in reg_list:
					#print reg
					if reg.rstrip() not in stopwords:
						if num == 0:
							wordList = reg
							num = 1
						else:
							wordList = wordList + ' ' + reg
		text.append(wordList)
		if lineNum%1000 == 0:
			print lineNum

	d = pd.DataFrame({'uid':data['uid'].values.T, 'qlist':text})
	d.to_csv('../feature/testQlist.csv', index = False)

#-- numpy array to pandas dataframe add columns name
def getColName(colNum, stri):
	LogInfo(str(colNum)+','+stri)
	colName = []
	for i in range(colNum):
		colName.append(stri + str(i))
	return colName

#-- get label from raw data
def getLabel():
	data = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	data1 = data[['uid','age']]
	data1.to_csv('../feature/raw/trainAge.csv', index = False)
	
	data2 = data[['uid','edu']]
	data2.to_csv('../feature/raw/trainEdu.csv', index = False)
	
	data3 = data[['uid','gender']]
	data3.to_csv('../feature/raw/trainGender.csv', index = False)

	data = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	data1 = data[['uid']]
	data1.to_csv('../feature/raw/test.csv', index = False)

#-- get pattern feature	
def getPatternFeature():
	pattern = pd.read_csv('../data/pattern.csv', header = 0)
	pattern = np.unique(pattern['pid'].values.T)
	print len(pattern)
	data = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	qlist = data['qlist'].values.T
	trainFeature = np.zeros([data.shape[0], len(pattern)])
	i = 0
	for line in qlist:
		j = 0
		if i%1000 == 1:
			print i
		listQuery = line.split(' ')
		for pa in pattern:
			for query in listQuery:
				if pa in query:
					trainFeature[i,j] += 1
			j += 1
		i += 1
	matrixColName = getColName(len(pattern), 'pattern')
	trainFeature = pd.DataFrame(trainFeature, columns = matrixColName)
	trainFeature['uid'] = data['uid'].values.T

	trainFeature.to_csv('../feature/basic/trainPatternFeature.csv', index = False)
	
	data = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	qlist = data['qlist'].values.T
	testFeature = np.zeros([data.shape[0], len(pattern)])
	i = 0
	for line in qlist:
		j = 0
		if i%1000 == 1:
			print i
		listQuery = line.split(' ')
		for pa in pattern:
			for query in listQuery:
				if pa in query:
					testFeature[i,j] += 1
			j += 1
		i += 1
	matrixColName = getColName(len(pattern), 'pattern')
	testFeature = pd.DataFrame(testFeature, columns = matrixColName)
	testFeature['uid'] = data['uid'].values.T

	testFeature.to_csv('../feature/basic/testPatternFeature.csv', index = False)

#-- get count feature
def getCountFeature():
	data = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	qlist = data['qlist'].values.T
	httpNum = []
	qNum = []
	qUniqueNum = []
	i = 0
	for line in qlist:
		i += 1
		if i%1000 == 1:
			print i
		listQuery = line.split(' ')
		qNum.append(len(listQuery))
		qUniqueNum.append(len(np.unique(listQuery)))
		
		results=re.findall("(?isu)(http\://[a-zA-Z0-9\.\?/&\=\:]+)", line)
		httpNum.append(len(results))
	
	feature = pd.DataFrame({'uid':data['uid'].values.T, 'httpNum':httpNum, 'qNum':qNum, 'qUniqueNum':qUniqueNum})
	feature.to_csv('../feature/basic/trainCountFeature.csv', index = False)
	
	data = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ',', encoding = 'utf-8')
	qlist = data['qlist'].values.T
	httpNum = []
	qNum = []
	qUniqueNum = []
	i = 0
	for line in qlist:
		i += 1
		if i%1000 == 1:
			print i
		listQuery = line.split(' ')
		qNum.append(len(listQuery))
		qUniqueNum.append(len(np.unique(listQuery)))
		results=re.findall("(?isu)(http\://[a-zA-Z0-9\.\?/&\=\:]+)", line)
		httpNum.append(len(results))

	feature = pd.DataFrame({'uid':data['uid'].values.T, 'httpNum':httpNum, 'qNum':qNum, 'qUniqueNum':qUniqueNum})
	feature.to_csv('../feature/basic/testCountFeature.csv', index = False)

#-- lsi model function
def lsi(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts))
	dictionary = corpora.Dictionary(texts)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get corpus..'
	corpusD = [dictionary.doc2bow(text) for text in texts]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'tfidf Model...'
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]

	model = LsiModel(corpusD, num_topics=topicNum, chunksize=8000, extra_samples = 100)#, distributed=True)#, sample = 1e-5, iter = 10,seed = 1)

	lsiFeature = np.zeros((len(texts), topicNum))
	print 'translate...'
	i = 0

	for doc in corpusD:
		topic = model[doc]
		
		for t in topic:
			 lsiFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i)

	return lsiFeature

#-- get lsi feature
def getLsiFeature(dim):
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ",")
	
	train = train[['uid','qlist']]
	test = test[['uid','qlist']]
	data = pd.concat([train, test], axis = 0)
	lsiFeature = lsi(data['qlist'].values, dim)
	colName = getColName(dim, "qlsi")
	lsiFeature = pd.DataFrame(lsiFeature, columns = colName)	
	lsiFeature['uid'] = data['uid'].values.T
	print lsiFeature.shape
	name = '../feature/lsi/lsiFeature'+str(dim)+'.csv'
	lsiFeature.to_csv(name, index = False)	

#-- get tf-idf model
def textTfidfEncodeModel():
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	data = pd.concat([train, test], axis = 0)
	
	documents =  train['qlist'].values
	encode = TfidfVectorizer(decode_error = 'ignore', norm = "l2", binary = False, sublinear_tf = True)
	result = encode.fit(documents)
	return result
 
#-- word2vec model function
def word2vec(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
	    for token in text:
	    	#token = int(token)
	    	#print token
	        frequency[token] += 1

	texts = [[token for token in text if frequency[token] >= 20] for text in texts]

	print 'train Model...'
	model = Word2Vec(texts, size=topicNum, window=5, iter = 15, min_count=20, workers=12, seed = 12)#, sample = 1e-5, iter = 10,seed = 1)
	path = '../feature/w2v/'+str(topicNum)+'w2vModel.m'
	model.save(path)
	w2vFeature = np.zeros((len(texts), topicNum))
	w2vFeatureAvg = np.zeros((len(texts), topicNum))
	
	i = 0
	for line in texts:
		num = 0
		for word in line:
			num += 1
			vec = model[word]
			w2vFeature[i, :] += vec
		w2vFeatureAvg[i,:] = w2vFeature[i,:]/num
		i += 1
		if i%5000 == 0:
			print i
	
	return w2vFeature, w2vFeatureAvg

#-- get word2vec feature
def getWord2vecFeature(dim):
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ",")
	train = train[['uid','qlist']]
	test = test[['uid','qlist']]
	data = pd.concat([train, test], axis = 0)
	
	vecFeature, vecFeatureAvg = word2vec(data['qlist'].values, dim)
	colName = getColName(dim, "vecT")
	vecFeature = pd.DataFrame(vecFeature, columns = colName)	
	
	colName = getColName(dim, "vecA")
	vecFeatureAvg = pd.DataFrame(vecFeatureAvg, columns = colName)	
	
	vecFeature['uid'] = data['uid'].values.T
	vecFeatureAvg['uid'] = data['uid'].values.T
	
	print vecFeature.shape, vecFeatureAvg.shape
	name = '../feature/w2v/w2vFeature'+str(dim)+'.csv'
	vecFeature.to_csv(name, index = False)

	name = '../feature/w2v/w2vFeatureAvg'+str(dim)+'.csv'
	vecFeatureAvg.to_csv(name, index = False)

#-- doc2vec model function
def doc2vec(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print len(texts)

	model = Doc2Vec(texts, size=topicNum, window=8,min_count=18, seed = 1)#, sample = 1e-5, iter = 10,seed = 1)

	doc2vecFeature = np.zeros((len(texts), topicNum))

	for i in range(len(texts)):
		
		vec = model.docvecs[i] 
		doc2vecFeature[i, :] = round(t[1],5)
	
	return doc2vecFeature
#-- get doc2vec feature
def getDoc2vecFeature(dim):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	data = pd.concat([train, test], axis = 0)
	
	vecFeature = doc2vec(data['qlist'].values, dim)
	colName = getColName(dim, "qvec")
	vecFeature = pd.DataFrame(vecFeature, columns = colName)	
	
	
	vecFeature['uid'] = data['uid'].values.T
	print vecFeature.shape
	name = '../feature/d2v/d2vFeature'+str(dim)+'.csv'
	vecFeature.to_csv(name, index = False)

#-- lda model function
def lda(documents, topicNum):
	texts = [[word for word in document.split(' ')] for document in documents]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(len(texts))
	dictionary = corpora.Dictionary(texts)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get corpus..'
	corpusD = [dictionary.doc2bow(text) for text in texts]

	#id2word = dictionary.id2word
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'tfidf Model...'
	tfidf = TfidfModel(corpusD)
	corpus_tfidf = tfidf[corpusD]
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'train lda Model...'
	ldaModel = gensim.models.ldamulticore.LdaMulticore(corpus_tfidf, workers = 8, num_topics=topicNum, chunksize=8000, passes=10, random_state = 12)
	#ldaModel = gensim.models.ldamodel.LdaModel(corpus=corpusD, num_topics=topicNum, update_every=1, chunksize=8000, passes=10)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get lda feature...'
	ldaFeature = np.zeros((len(texts), topicNum))
	i = 0

	for doc in corpus_tfidf:
		topic = ldaModel.get_document_topics(doc, minimum_probability = 0.01)
		
		for t in topic:
			 ldaFeature[i, t[0]] = round(t[1],5)
		i = i + 1
		if i%1000 == 1:
			print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(i)

	return ldaFeature

#-- get lda feature function
def getLdaFeature(dim):
	#---user Lda info--
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0, sep = ",")

	train = train[['uid','qlist']]
	test = test[['uid','qlist']]
	data = pd.concat([train, test], axis = 0)
	ldaFeature = lda(data['qlist'].values, dim)
	colName = getColName(dim, "qlda")
	ldaFeature = pd.DataFrame(ldaFeature, columns = colName)	
	ldaFeature['uid'] = data['uid'].values.T
	print ldaFeature.shape
	name = '../feature/lda/ldaFeature'+str(dim)+'.csv'
	ldaFeature.to_csv(name, index = False)

#-- read feature 	
def readFeature(config):
	train1 = pd.read_csv('../feature/raw/trainAge.csv', header = 0)
	train2 = pd.read_csv('../feature/raw/trainEdu.csv', header = 0)
	train3 = pd.read_csv('../feature/raw/trainGender.csv', header = 0)
	train = pd.merge(train1, train2, on = 'uid', how = 'left')
	train = pd.merge(train, train3, on = 'uid', how = 'left')
	
	labelName = ['age','edu','gender']
	test = pd.read_csv('../feature/raw/test.csv', header = 0)
	LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	if config['matrixStackFeature'] == True:
		pathList = config['pathList']
		for path in pathList:
			data = pd.read_csv('../feature/stack/'+path+'.csv', header = 0)
			data = data.values
			data1 = pd.DataFrame(data[0:100000])
			data2 = pd.DataFrame(data[100000:200000])
			
			train = pd.concat([train, data1], axis = 1)
			test = pd.concat([test, data2], axis = 1)
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')


	if config['lsiFeature'] == True:
		for l in config['lsiList']:
			name = '../feature/lsi/lsiFeature'+str(l)+'.csv'
			data = pd.read_csv(name, header = 0)
		
			train = pd.merge(train, data, on = 'uid', how = 'left')
			test = pd.merge(test, data, on = 'uid', how = 'left')
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	if config['countFeature'] == True:
		trainData = pd.read_csv('../feature/basic/trainCountFeature.csv', header = 0)
		testData = pd.read_csv('../feature/basic/testCountFeature.csv', header = 0)

		train = pd.merge(train, trainData, on = 'uid', how = 'left')
		test = pd.merge(test, testData, on = 'uid', how = 'left')
		LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	if config['myStackFeature'] == True:
		pathList = config['pathList2']
		for path in pathList:
			data1 = pd.read_csv('../feature/stack-my/'+path+'_train_prob.csv', header = 0)
			data2 = pd.read_csv('../feature/stack-my/'+path+'_test_prob.csv', header = 0)
			
			train = pd.concat([train, data1], axis = 1)
			test = pd.concat([test, data2], axis = 1)
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	if config['patternFeature'] == True:
		trainData = pd.read_csv('../feature/basic/trainPatternFeature.csv', header = 0)
		testData = pd.read_csv('../feature/basic/testPatternFeature.csv', header = 0)

		train = pd.merge(train, trainData, on = 'uid', how = 'left')
		test = pd.merge(test, testData, on = 'uid', how = 'left')
		LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	if config['ldaFeature'] == True:
		for l in config['ldaList']:
			name = '../feature/lda/ldaFeature'+str(l)+'.csv'
			data = pd.read_csv(name, header = 0)
		
			train = pd.merge(train, data, on = 'uid', how = 'left')
			test = pd.merge(test, data, on = 'uid', how = 'left')
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')
	if config['d2vFeature'] == True:
		for l in config['d2vList']:
			name = '../feature/d2v/d2vFeature'+str(l)+'.csv'
			data = pd.read_csv(name, header = 0)
		
			train = pd.merge(train, data, on = 'uid', how = 'left')
			test = pd.merge(test, data, on = 'uid', how = 'left')
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')
	if config['w2vFeature'] == True:
		for l in config['w2vList']:
			name = '../feature/w2v/w2vFeatureAvg'+str(l)+'.csv'
			data = pd.read_csv(name, header = 0)
		
			train = pd.merge(train, data, on = 'uid', how = 'left')
			test = pd.merge(test, data, on = 'uid', how = 'left')
			LogInfo('['+str(train.shape[0])+","+str(train.shape[1])+'],['+str(test.shape[0])+","+str(test.shape[1])+']')

	

	
	#print train.columns
	testUid = test['uid'].values.T
	trainUid = train['uid'].values.T
	trainLabelAge = train['age'].values.T
	trainLabelGender = train['gender'].values.T
	trainLabelEdu = train['edu'].values.T


	# deleteName = []
	deleteName = labelName
	deleteName.append('uid')
	print deleteName
	trainFeature = train.drop(deleteName, axis = 1)
	testFeature = test.drop(['uid'], axis = 1)

	trainFeature = trainFeature.fillna(0)
	testFeature = testFeature.fillna(0)

	if config['usePCA'] == True:
		pca = IncrementalPCA(n_components=200)
		data = pd.concat([trainFeature, testFeature], axis = 0)
		print 'pca data shape: '
		print data.shape
		pca.fit(data.fillna(0))

		trainFeature = pca.transform(trainFeature.fillna(0))
		testFeature = pca.transform(testFeature.fillna(0))
		print trainFeature.shape, testFeature.shape, len(trainLabel)

		return trainFeature, testFeature, trainLabel, trainUid, testUid



	print trainFeature.shape, testFeature.shape, len(trainLabelAge)

	return trainFeature.values, testFeature.values, trainLabelAge, trainLabelGender, trainLabelEdu


	








