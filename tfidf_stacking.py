# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# import jieba as jb

train = pd.read_csv(r'../feature/raw/trainQlist.csv', encoding='utf-8')
test = pd.read_csv(r'../feature/raw/testQlist.csv', encoding='utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec = TfidfVectorizer(min_df=1, norm='l2',use_idf=True, sublinear_tf=True, smooth_idf=True)
tf_vec.fit(pd.concat([train.qlist, test.qlist], axis=0))
X_train_tf = tf_vec.transform(train.qlist)
X_test_tf = tf_vec.transform(test.qlist)


# # SKlearn classification models
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import RidgeClassifier

#cross validation
from scipy import sparse
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
# skf = StratifiedKFold(train.label, n_folds=9, shuffle=False, random_state=42)
# x = train_x
# y = train.edu

#Naive Bayes models
mnb_clf = MultinomialNB()
gnb_clf = GaussianNB()
bnb_clf = BernoulliNB()

#SVM_based models
sgd_clf = SGDClassifier(loss = 'hinge',penalty = 'l2', alpha = 0.0001,n_iter = 500, random_state = 42, verbose=1, n_jobs=256)

svc_clf = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.000001, C=0.5, multi_class='ovr', 
                         fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=1, random_state=None, max_iter=5000)

svm_clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True, 
                  tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape=None, random_state=None)

#Logistic Regression
lr_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                            class_weight=None, random_state=None, solver='liblinear', max_iter=5000, 
                            multi_class='ovr', verbose=1, warm_start=False, n_jobs=256)


#creat lr stacking features
#gender
from sklearn.feature_selection import SelectKBest, chi2, f_classif
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.gender)
x_test = ch2.transform(X_test_tf)

print 'creat gender prob features'
random_seed = 2016
x = x_train
y = [1 if i == 1 else 0 for i in train.gender]
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,1))
new_test = np.zeros((100000,1))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.gender[trainid]
    val_x = x_train[valid]
    lr_clf.fit(train_x, train_y)
    new_train[valid] = lr_clf.predict_proba(val_x)[0]
    new_test += lr_clf.predict_proba(x_test)[0]
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('lr_gender_',i) for i in range(3)]
stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#age
print 'creat age prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.age)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.age
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.age[trainid]
    val_x = x_train[valid]
    lr_clf.fit(train_x, train_y)
    new_train[valid] = lr_clf.predict_proba(val_x)[0]
    new_test += lr_clf.predict_proba(x_test)[0]
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('lr_age_',i) for i in range(7)]
stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#edu
print 'creat edu prob features'

ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.edu)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.edu
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.edu[trainid]
    val_x = x_train[valid]
    lr_clf.fit(train_x, train_y)
    new_train[valid] = lr_clf.predict_proba(val_x)[0]
    new_test += lr_clf.predict_proba(x_test)[0]
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('lr_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
lr_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
lr_prob_feat.to_csv(r'../feature/stack/lr_prob.csv', index=0)

#creat linearsvc stacking features

#gender
print 'creat gender prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.gender)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.gender

skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,3))
new_test = np.zeros((100000,3))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.gender[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('svc_gender_',i) for i in range(3)]
stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#age
print 'creat age prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.age)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.age
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.age[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('svc_age_',i) for i in range(7)]
stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#edu
print 'creat edu prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.age)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.edu
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.edu[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('svc_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
svc_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
svc_prob_feat.to_csv(r'../feature/stack/svc_prob.csv', index=0)

#creat bnb stacking features
#gender
print 'creat gender prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.gender)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.gender
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,3))
new_test = np.zeros((100000,3))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.gender[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('bnb_gender_',i) for i in range(3)]
stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#age
print 'creat age prob features'
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.age)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.age
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.age[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('bnb_age_',i) for i in range(7)]
stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#edu
ch2 = SelectKBest(chi2, k=100000)
x_train = ch2.fit_transform(X_train_tf, train.edu)
x_test = ch2.transform(X_test_tf)

random_seed = 2016
x = x_train
y = train.edu
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = x_train[trainid]
    train_y = train.edu[trainid]
    val_x = x_train[valid]
    svc_clf.fit(train_x, train_y)
    new_train[valid] = svc_clf.decision_function(val_x)
    new_test += svc_clf.decision_function(x_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('bnb_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
bnb_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
bnb_prob_feat.to_csv(r'../feature/stack/bnb_prob.csv', index=0)

#creat mnb stacking features
#tfidf min_df=50
#gender
tf_vec = TfidfVectorizer(min_df=50, norm='l2',use_idf=True, sublinear_tf=True, smooth_idf=True)
tf_vec.fit(pd.concat([train.qlist, test.qlist], axis=0))
X_train_tf = tf_vec.transform(train.qlist)
X_test_tf = tf_vec.transform(test.qlist)

print 'creat gender prob features'
random_seed = 2016
x = X_train_tf
y = train.gender
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,3))
new_test = np.zeros((100000,3))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train_tf[trainid]
    train_y = train.gender[trainid]
    val_x = X_train_tf[valid]
    mnb_clf.fit(train_x, train_y)
    new_train[valid] = mnb_clf.predict_proba(val_x)
    new_test += mnb_clf.predict_proba(X_test_tf)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('mnb_gender_',i) for i in range(3)]
stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#age
print 'creat age prob features'
random_seed = 2016
x = X_train_tf
y = train.age
skf = StratifiedKFold(y, n_folds=5, shuffle=True)


new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train_tf[trainid]
    train_y = train.age[trainid]
    val_x = X_train_tf[valid]
    mnb_clf.fit(train_x, train_y)
    new_train[valid] = mnb_clf.predict_proba(val_x)
    new_test += mnb_clf.predict_proba(X_test_tf)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('mnb_age_',i) for i in range(7)]
stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#edu
print 'creat edu prob features'
random_seed = 2016
x = X_train_tf
y = train.edu
skf = StratifiedKFold(y, n_folds=5, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train_tf[trainid]
    train_y = train.edu[trainid]
    val_x = X_train_tf[valid]
    mnb_clf.fit(train_x, train_y)
    new_train[valid] = mnb_clf.predict_proba(val_x)
    new_test += mnb_clf.predict_proba(X_test_tf)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('mnb_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
mnb_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
mnb_prob_feat.to_csv(r'../feature/stack/mnb_prob.csv', index=0)