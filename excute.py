# -- encoding:UTF-8 --
#-- Author: TNT_000 by Abner yang
import numpy as np 
import pandas as pd 
import jieba
import jieba.posseg
import jieba.analyse
import re
from collections import defaultdict
from function import *
from model import *
#-- xgb parameters
params={
	'booster':'gbtree',
	'objective': 'multi:softmax',
	#'objective': 'rank:pairwise',
    'eval_metric': 'merror',
	'stratified':True,

	'max_depth':8,
	'min_child_weight':0.01,
	'gamma':0.1,
	'subsample':0.6,
	'colsample_bytree':0.5,
	#'max_delta_step':1,
	#'colsample_bylevel':0.5,
	#'rate_drop':0.3,
	
	'lambda':0.0001,   #550
	'alpha':10,
	#'lambda_bias':0,
	
	'eta': 0.02,
	'seed':12,
	'nthread':24,
	'silent':1
}

#-- config info 
config = {
	'params':params,
	'usePCA':False, # True: use PCA solution False: no use
	'ldaDim':np.arange(10,91,5), # get lda feature list
	'w2vDim':np.arange(500,501,10), # get word2vec feature list
	'lsiDim':np.arange(100,101,10), # get lsi feature list
	'roundsAge':800, #-- Age problem xgb train round
	'roundsGender':700,#-- Gender problem xgb train round
	'roundsEdu':850,#-- Edu problem xgb train round
	'folds':3, #-- cross validation folds
	'getStack':False, 
	'stackFeature':False,
	'multiClass':True, #-- True: multi class  False: binary class
	'countFeature':True, #-- True: read count feature False no read
	'patternFeature':True, #-- True: read pattern feature False no read
	'ldaFeature':True, #-- True: read lda feature False no read
	'lsiFeature':False, #-- True: read lsi feature False no read
	'w2vFeature':True, #-- True: read w2v feature False no read
	'd2vFeature':True, #-- True: read d2v feature False no read
	'fillLabel':False,  #-- True: fill 0 label with fillAge or fillGender or fillEdu parameter False: no fill
	'fillAge':1,
	'fillGender':1,
	'fillEdu':5,
	'matrixStackFeature':True, #-- True: read teamate's stack feature False no read
	'myStackFeature':False, #-- True: read my stack feature False no read
	'ldaList':np.arange(10,100,5), # read lda feature list
	'd2vList':np.arange(300,301,100), # read doc2vec feature list
	'lsiList':np.arange(100,101,10), # read lsi feature list
	'w2vList':np.arange(100,101,10), # read word2vec feature list
	'stackList':[1,3,4],
	#'pathList':['cnn_prob']#,'ft_prob']#
	'pathList':['lr_prob_10','mnb_prob','bnb_prob_10','svc_prob_10','lr_prob','bnb_prob','svc_prob'],#-- read stack feature name list
	'modelName':['rf-15','rf-16','etc-13','etc-14''etc-15','etc-16','etc-17','etc-18','lr-11'], #-- stack model name list
	'taskList':['age','gender','edu']# task name list for stack
}

if __name__ == '__main__':
	#-- pre process raw data
	getWord()
	getLabel()
	#-- get feature from raw data
	getCountFeature()
	getPatternFeature()
	getLsiFeature()
	getLdaFeature()
	getWord2vecFeature()
	getDoc2vecFeature()
	#--- read feature and label
	trainFeature, testFeature, trainLabelAge, trainLabelGender, trainLabelEdu = readFeature(config)
	#--- cross vaidation with problem age
	config['task'] = 'age'
	params = paramsAge
	rounds = config['roundsAge']
	res = xgbCVModel(trainFeature, trainLabelAge, rounds, config['folds'], config, params)
	
	#--- cross vaidation with problem gender
	config['task'] = 'gender'
	params = paramsGender
	rounds = config['roundsGender']
	res = xgbCVModel(trainFeature, trainLabelGender, rounds, config['folds'], config, params)
	
	#--- cross vaidation with problem edu
	config['task'] = 'edu'
	params = paramsEdu
	rounds = config['roundsEdu']
	res = xgbCVModel(trainFeature, trainLabel, rounds, config['folds'], config, params)

	#--- online predict 
	config['prob'] = False
	config['params'] = paramsEdu
	excutePredictModel('edu', 'xgb', config)
	
	config['prob'] = False
	config['params'] = paramsGender
	excutePredictModel('gender', 'xgb', config)
	
	config['prob'] = False
	config['params'] = paramsAge
	excutePredictModel('age', 'xgb', config)
	#--- get online predict result
	mergeResult('2016-12-07-xgb-01')
	

	#--- get stack feature  by clf model list
	clfs = [RandomForestClassifier(n_estimators=900, n_jobs=-1, criterion='gini'),
			RandomForestClassifier(n_estimators=900, n_jobs=-1, criterion='entropy'),
			ExtraTreesClassifier(n_estimators=800, n_jobs=-1, criterion='gini'),
			ExtraTreesClassifier(n_estimators=800, n_jobs=-1, criterion='entropy'),
			ExtraTreesClassifier(n_estimators=900, n_jobs=-1, criterion='gini'),
			ExtraTreesClassifier(n_estimators=900, n_jobs=-1, criterion='entropy'),
			ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
			ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
			LogisticRegression(solver='liblinear', n_jobs=8)]
			GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

	LogInfo("Creating train and test sets for stack.")
	data = {} 
	data['trainFeature'] = trainFeature
	data['trainLabel1'] = trainLabelAge
	data['trainLabel2'] = trainLabelGender
	data['trainLabel3'] = trainLabelEdu
	data['testFeature'] = testFeature
	stackFrame(data, config, clfs)

	

