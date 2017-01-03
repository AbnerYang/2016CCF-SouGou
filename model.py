# -- encoding:UTF-8 --
#-- Author: TNT_000 by Abner yang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np 
import pandas as pd 
import xgboost as xgb 
from sklearn.preprocessing import OneHotEncoder
from function import *
from scipy.sparse import hstack
from matplotlib import pyplot
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import *
from keras.models import Sequential
from keras.optimizers import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

#-- eval function
def error(true, predict):
	print true[0:5]
	print predict[0:5]
	k = true - predict
	return 1 - float(len(k[k == 0]))/len(k)
#-- svm local cross validation model frame
def svmLocalCVModel(trainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	trainUid = train['uid'].values.T
	print "Train tf-idf vector Model..."	
	encode = textTfidfEncodeModel()
	print 'CV On SVC Model....'
  	kfold = StratifiedKFold(y = trainLabel, n_folds = 3, shuffle = True, random_state = 12)
  	f = 0
  	predict = []
  	true = []
  	uid = []
  	for index1, index2 in kfold:
  		#print index2
  		print 'fold:'+str(f)
  		#encode = TfidfVectorizer(decode_error = 'ignore', binary = False, sublinear_tf = True)
  		localTrainFeature = encode.transform(train['qlist'].values[index1])
  		localTestFeature = encode.transform(train['qlist'].values[index2])
  		localTrainLabel = trainLabel[index1]
  		localTestLabel = trainLabel[index2]
  		localTestUid = trainUid[index2]
  		uid =np.append(uid, localTestUid)
  		print localTrainFeature.shape
  		print 'Build, Train and Predict SVC Model.....'
  		#print localTrainFeature.shape[1]
  		localPredict = svmLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config)  		
  		if config['prob'] == True:
  			if f == 0:
  				predict = localPredict
  			else:
  				predict = np.concatenate((predict, localPredict), axis = 0)
  		else:
  			print error(localTestLabel, localPredict)
  			predict = np.append(predict,localPredict)
  			true = np.append(true, localTestLabel)
  		f += 1
  	if config['prob'] == True:
  		return predict, uid
  	else:
  		print "Total error"+str(error(true, predict))
  		return predict, uid

#-- svm local train-test model frame
def svmLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config):
	print 'train...'
	if config['prob'] == True:
		model = SVC(kernel = 'linear', cache_size = 200, random_state = 12, probability = True)
  	else:
  		model = SVC(kernel = 'linear', cache_size = 200, random_state = 12, probability = False)
  	model.fit(X = localTrainFeature, y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == True:
  		return model.predict_log_proba(localTestFeature)
  	else:
  		return model.predict(localTestFeature)

#-- svm local online predict model frame
def svmPredictModel(localTrainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	print "Train tf-idf vector Model..."	
	encode = textTfidfEncodeModel()
	
	localTrainFeature = encode.transform(train['qlist'].values)
	localTestFeature = encode.transform(test['qlist'].values)

	print localTrainFeature.shape, localTestFeature.shape

	print 'train...'
	if config['prob'] == True:
		model = SVC(kernel = 'linear', cache_size = 200, random_state = 12, probability = True)
  	else:
  		model = SVC(kernel = 'linear', cache_size = 200, random_state = 12, probability = False)
  	
	model.fit(X = localTrainFeature, y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == True:
  		return model.predict_log_proba(localTestFeature), test['uid'].values.T
  	else:
  		return model.predict(localTestFeature), test['uid'].values.T

#-- Gaussian Navie Bayes local cross validation model frame
def GaussianNBLocalCVModel(trainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	trainUid = train['uid'].values.T
	print "Train tf-idf vector Model..."	
	documents =  train['qlist'].values
	encode = TfidfVectorizer( norm = "l1", binary = False, sublinear_tf = True, min_df = 5, max_df = 0.9)
	encode.fit(documents)
	print 'CV On GNB Model....'
  	kfold = StratifiedKFold(y = trainLabel, n_folds = 3, shuffle = True, random_state = 12)
  	f = 0
  	predict = []
  	true = []
  	uid = []
  	for index1, index2 in kfold:
  		#print index2
  		print 'fold:'+str(f)
  		localTrainFeature = encode.transform(train['qlist'].values[index1])
  		localTestFeature = encode.transform(train['qlist'].values[index2])
  		localTrainLabel = trainLabel[index1]
  		localTestLabel = trainLabel[index2]
  		localTestUid = trainUid[index2]
  		uid = np.append(uid, localTestUid)
  		print localTrainFeature.shape
  		print 'Build, Train and Predict GaussianNB Model.....'
  		#print localTrainFeature.shape[1]
  		localPredict = GaussianNBLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config)
  		if config['prob'] == True:
  			if f == 0:
  				predict = localPredict
  			else:
  				predict = np.concatenate((predict, localPredict), axis = 0)
  		else:
  			print error(localTestLabel, localPredict)
  			predict = np.append(predict,localPredict)
  			true = np.append(true, localTestLabel)
  		f += 1
  	if config['prob'] == True:
  		return predict, uid
  	else:
  		print "Total error"+str(error(true, predict))
  		return predict, uid

#-- Gaussian Navie Bayes train-test model frame
def GaussianNBLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config):
	print 'train...'
	model = GaussianNB()
  	model.fit(X = localTrainFeature.toarray(), y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == False:
  		return model.predict(localTestFeature.toarray())
  	else:
  		return model.predict_log_proba(localTestFeature.toarray())

#-- Gaussian Navie Bayes online predict model frame
def GaussianNBPredictModel(localTrainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	print "Train tf-idf vector Model..."
	encode = TfidfVectorizer(decode_error = 'ignore', norm = "l2", binary = False, sublinear_tf = True, min_df = 50)
	localTrainFeature = encode.fit_transform(train['qlist'].values)
	localTestFeature = encode.transform(train['qlist'].values)

	print localTrainFeature.shape, localTestFeature.shape

	print 'train...'
	model = GaussianNB()
  	model.fit(X = localTrainFeature.toarray(), y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == False:
  		return model.predict(localTestFeature.toarray()), test['uid'].values
  	else:
  		return model.predict_log_proba(localTestFeature.toarray()), test['uid'].values

#-- Multinomial Navie Bayes corss validation model frame
def MultinomialNBLocalCVModel(trainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	trainUid = train['uid'].values.T
	print "Train tf-idf vector Model..."	
	documents =  train['qlist'].values
	encode = TfidfVectorizer(decode_error = 'ignore', norm = "l2", binary = False, sublinear_tf = True, min_df = 50)
	encode.fit(documents)
	print 'CV On SVC Model....'
  	kfold = StratifiedKFold(y = trainLabel, n_folds = 3, shuffle = True, random_state = 12)
  	f = 0
  	predict = []
  	true = []
  	uid = []
  	for index1, index2 in kfold:
  		#print index2
  		print 'fold:'+str(f)
  		localTrainFeature = encode.transform(train['qlist'].values[index1])
  		localTestFeature = encode.transform(train['qlist'].values[index2])
  		localTrainLabel = trainLabel[index1]
  		localTestLabel = trainLabel[index2]
  		localTestUid = trainUid[index2]
  		uid = np.append(uid, localTestUid)
  		print localTrainFeature.shape
  		print 'Build, Train and Predict MultinomialNB Model.....'
  		#print localTrainFeature.shape[1]
  		localPredict = MultinomialNBLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config)
  		if config['prob'] == True:
  			if f == 0:
  				predict = localPredict
  			else:
  				predict = np.concatenate((predict, localPredict), axis = 0)
  		else:
  			print error(localTestLabel, localPredict)
  			predict = np.append(predict,localPredict)
  			true = np.append(true, localTestLabel)
  		f += 1
  	if config['prob'] == True:
  		return predict, uid
  	else:
  		print "Total error"+str(error(true, predict))
  		return predict, uid

#-- Multinomial Navie Bayes train-test model frame
def MultinomialNBLocalModel(localTrainFeature, localTestFeature, localTrainLabel, config):
	print 'train...'
	model = MultinomialNB(alpha=1, fit_prior=True, class_prior=None)
  	model.fit(X = localTrainFeature, y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == False:
  		return model.predict(localTestFeature)
  	else:
  		return model.predict_log_proba(localTestFeature)

#-- Multinomial Navie Bayes online predict model frame
def MultinomialNBPredictModel(localTrainLabel, config):
	train = pd.read_csv('../feature/trainQlist.csv', header = 0, sep = ",")
	test = pd.read_csv('../feature/testQlist.csv', header = 0, sep = ",")
	print "Train tf-idf vector Model..."	
	encode = TfidfVectorizer(decode_error = 'ignore', norm = "l2", binary = False, sublinear_tf = True, min_df = 50)
	localTrainFeature = encode.fit_transform(train['qlist'].values)
	localTestFeature = encode.transform(train['qlist'].values)

	print localTrainFeature.shape, localTestFeature.shape

	print 'train...'
	model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
  	model.fit(X = localTrainFeature, y = localTrainLabel)
  	print 'predict...'
  	if config['prob'] == False:
  		return model.predict(localTestFeature), test['uid'].values
  	else:
  		return model.predict_log_proba(localTestFeature), test['uid'].values

#-- xgboost local corss validation model frame
def xgbLocalCVModel(taskName, config):
	params = config['params']
	config['task'] = taskName
	trainFeature, testFeature, trainLabel, trainUid, testUid = readFeature(config)
	if taskName == 'gender':
		rounds = config['roundsGender']
	elif taskName == 'age':
		rounds = config['roundsAge']
	else:
		rounds = config['roundsEdu']

	if config['multiClass'] == True:
		params['num_class'] = len(np.unique(trainLabel))
		print params['num_class']
	else:
		params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
		print params['scale_pos_weight']

	if config['prob'] == True:
		params['objective'] = 'multi:softprob'
	else:
		params['objective'] = 'multi:softmax'
	
	print 'CV On XGB Model....'
  	kfold = StratifiedKFold(y = trainLabel, n_folds = config['folds'], shuffle = True, random_state = params['seed'])
  	f = 0
  	predict = []
  	true = []
  	uid = []
  	for index1, index2 in kfold:
  		print 'fold:'+str(f)
  		print index1, index2
  		localTrainFeature = trainFeature[index1,:]
  		localTestFeature = trainFeature[index2,:]
  		localTrainLabel = trainLabel[index1]
  		localTestLabel = trainLabel[index2]
  		localTestUid = trainUid[index2]
  		uid = np.append(uid, localTestUid)
  		print 'Build, Train and Predict XGB Model.....'
  		#print localTrainFeature.shape[1]
  		localPredict = xgbLocalModel(localTrainFeature, localTestFeature, localTrainLabel, localTestLabel, params, config, rounds)
  		
  		if config['prob'] == True:
  			if f == 0:
  				predict = localPredict
  			else:
  				predict = np.concatenate((predict, localPredict), axis = 0)
  		else:
  			print error(localTestLabel, localPredict)
  			predict = np.append(predict,localPredict)
  			true = np.append(true, localTestLabel)
  		f += 1
  	if config['prob'] == True:
  		return predict, uid
  	else:
  		print "Total error"+str(error(true, predict))
  		return predict, uid

#-- xgboost local train-test model frame
def xgbLocalModel(trainFeature, testFeature, trainLabel, testLabel, params, config, rounds):
	
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = testLabel)

	watchlist  = [(dtest,'eval'), (dtrain,'train')]

	num_round = rounds
	
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 10)
	
	pred = model.predict(dtest)

	return pred

#-- xgboost online predict model frame
def xgbPredictModel(taskName, config):

	params = config['params']
	config['task'] = taskName
	trainFeature, testFeature, trainLabelAge, trainLabelGender, trainLabelEdu = readFeature(config)
	if taskName == "gender":
		rounds = config['roundsGender']
		trainLabel = trainLabelGender
	elif taskName == 'age':
		rounds = config['roundsAge']
		trainLabel = trainLabelAge
	else:
		rounds = config['roundsEdu']
		trainLabel = trainLabelEdu

	if config['prob'] == True:
		params['objective'] = 'multi:softprob'
	else:
		params['objective'] = 'multi:softmax'
	
	print np.unique(trainLabel)

	if config['multiClass'] == True:
		params['num_class'] = len(np.unique(trainLabel))
		print params['num_class']
	else:
		params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
		print params['scale_pos_weight']

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature)

	watchlist  = [(dtrain,'train')]
	
	#params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])

	#print params['scale_pos_weight']

	num_round = rounds
	
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 100)


	importance = pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort('importance', ascending=False)


	predict = model.predict(dtest)
	name = '../importance/im_'+config['task']+'.csv'
	name2 = '../model/xgb'+config['task']+'.m'
	importance.to_csv(name, index = False)
	model.save_model(name2)
	if config['prob'] == False:
		if config['fillLabel'] == True:
			predict = [int(p)+1 for p in predict]
		else:
			predict = [int(p) for p in predict]
	else:
		return predict
	
	return predict

#-- local cross validation excute frame work
def excuteLocalCVModel(taskName,core,config):

	if taskName == 'edu':
		print 'edu...'
		train = pd.read_csv('../feature/trainEdu.csv', header = 0)
		trainLabel = train['edu'].values.T
	elif taskName == 'gender':
		print 'gender...'
		train = pd.read_csv('../feature/trainGender.csv', header = 0)
		trainLabel = train['gender'].values.T
	else:
		print 'age...'
		train = pd.read_csv('../feature/trainAge.csv', header = 0)
		trainLabel = train['age'].values.T

	if core == 'svc':
		predict, uid = svmLocalCVModel(trainLabel, config)
	elif core == 'GNB':
		predict, uid = GaussianNBLocalCVModel(trainLabel, config)
	elif core == 'MNB':
		predict, uid = MultinomialNBLocalCVModel(trainLabel, config)
	else:
		predict, uid = xgbLocalCVModel(taskName,config)

	if config['prob'] == True:
		colname = getColName(predict.shape[1], "stack-"+core+"-"+taskName+"-prob")
		result = pd.DataFrame(predict, columns = colname)
		result['uid'] = uid
		result.to_csv('../feature/stack/stack-'+core+'-'+taskName+'-train-prob.csv',index = False)
	else:
		colname = getColName(1, "stack-"+core+"-"+taskName+"-val")
		result = pd.DataFrame(predict, columns = colname)
		result['uid'] = uid
		result.to_csv('../feature/stack/stack-'+core+'-'+taskName+'-train-val.csv',index = False)

#-- online predict excute frame work
def excutePredictModel(config):
	core = 'xgb'
	test = pd.read_csv('../feature/raw/test.csv',header = 0)
	uid = test['uid'].values.T
	for taskName in config['taskList']:
		print taskName

		predict = xgbPredictModel(taskName,config)
	
		colname = getColName(1, "stack-"+core+"-"+taskName+"-val")
		result = pd.DataFrame(predict, columns = colname)
		result['uid'] = uid
		result.to_csv('../feature/stack/stack-'+core+'-'+taskName+'-test-val.csv',index = False)

#-- get online predict result
def mergeResult(path):
	r1 = pd.read_csv('../feature/stack/stack-xgb-gender-test-val.csv', header = 0)
	r2 = pd.read_csv('../feature/stack/stack-xgb-age-test-val.csv', header = 0)
	r3 = pd.read_csv('../feature/stack/stack-xgb-edu-test-val.csv', header = 0)

	print 'age..'
	print np.unique(r2['stack-xgb-age-val0'].values.T)
	print 'gender..'
	print np.unique(r1['stack-xgb-gender-val0'].values.T)
	print 'edu..'
	print np.unique(r3['stack-xgb-edu-val0'].values.T)

	predict1 = r1['stack-xgb-gender-val0'].values.T
	predict1[predict1 == 0] = 1
	
	predict2 = r2['stack-xgb-age-val0'].values.T
	predict2[predict2 == 0] = 1

	predict3 = r3['stack-xgb-edu-val0'].values.T
	predict3[predict3 == 0] = 4

	print 'clear zero!'
	print 'gender..'
	print np.unique(predict1)
	print 'age..'
	print np.unique(predict2)
	print 'edu..'
	print np.unique(predict3)
	
	result = pd.DataFrame({'uid':r1['uid'].values.T, 'age':predict2, 'gender':predict1, 'edu':predict3})

	result = result[['uid','age','gender','edu']]
	print result.shape
	result.to_csv('../result/'+path+'.csv', index = False, header = False, sep = ' ')

#-- xgboost cross validation model frame
def xgbCVModel(trainFeature, trainLabel, rounds, folds, config, params):
	print trainFeature.shape, len(trainLabel)
	#trainFeature = sparse.csr_matrix(trainFeature)
	#testFeature = sparse.csr_matrix(testFeature)
	if config['fillLabel'] == True:
		trainLabel = np.array(trainLabel)
		trainFeature = trainFeature[trainLabel != 0,:]
		trainLabel = trainLabel[trainLabel != 0]
		trainLabel = trainLabel - 1
	print trainFeature.shape, len(trainLabel)
	

	print np.unique(trainLabel)
	#--Set parameter: scale_pos_weight-- 
	if config['multiClass'] == True:
		params['num_class'] = len(np.unique(trainLabel))
		print params['num_class']
	else:
		params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
		print params['scale_pos_weight']

	#--Get User-define DMatrix: dtrain--
	#print trainQid[0]
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	num_round = rounds

	#--Run CrossValidation--
	print 'run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds) 
	res = xgb.cv(params, dtrain, num_round, nfold = folds, verbose_eval = 1)
	return res

def evalerror(pred, true):
	er = 0
	for k in range(len(pred)):
		if pred[k] != true[k]:
			er += 1
	score = float(er)/len(pred)
	LogInfo(str(score))
	return score

#-- stack frame work
def stackFrame(data, config, clf_List):
	# -- get train /test feature and train label
	trainFeature = data['trainFeature']
	testFeature = data['testFeature']

	# -- get stack param from config
	cvfolds = config['folds']
	# -- stack train and test
	for j, clf in enumerate(clf_List):
		modelName = config['modelName'][j]
		LogInfo("Model-"+modelName)
		for labelIndex in range(3):
			labelName = 'trainLabel'+str(labelIndex+1)
			LogInfo(labelName)
			trainLabel = data[labelName]
			skf = list(StratifiedKFold(trainLabel, cvfolds))
			config['task'] = config['taskList'][labelIndex]
			# -- define the stack model result
			blend_train = np.zeros((trainFeature.shape[0], len(np.unique(trainLabel))))
			blend_test = np.zeros((testFeature.shape[0], len(np.unique(trainLabel))))
			for i, (trainIndex, testIndex) in enumerate(skf):
				LogInfo("Fold-"+str(i))
				X_train = trainFeature[trainIndex]
				y_train = trainLabel[trainIndex]
				X_test = trainFeature[testIndex]
				y_test = trainLabel[testIndex]

				if clf == 'xgb':
					y_pred, test_pred = xgbStackModel(X_train, X_test, y_train, y_test, testFeature, config)
					blend_test += test_pred
				else:
					clf.fit(X_train,y_train)
					y_pred = clf.predict_proba(X_test)
					y_pred_val = clf.predict(X_test)
					test_pred = clf.predict_proba(testFeature)
					blend_test += test_pred
					evalerror(y_pred_val, y_test)

				blend_train[testIndex,:] = y_pred
			
			blend_test = blend_test/cvfolds

			if labelIndex == 0:
				train = blend_train
				test = blend_test
			else:
				train = np.concatenate([train, blend_train], axis = 1)
				test = np.concatenate([test, blend_test], axis = 1)

		train = pd.DataFrame(train, columns = getColName(train.shape[1], modelName))
		test = pd.DataFrame(test, columns = getColName(test.shape[1], modelName))

		train.to_csv('../feature/stack-my/'+modelName+'_train_prob.csv', index = False)
		test.to_csv('../feature/stack-my/'+modelName+'_test_prob.csv', index = False)
