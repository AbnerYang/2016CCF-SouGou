# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import jieba as jb
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D

train = pd.read_csv(r'../feature/raw/trainQlist.csv')
test = pd.read_csv(r'../feature/raw/testQlist.csv')

corpus = pd.concat([train.qlist, test.qlist])
corpus.index = range(corpus.shape[0])

texts = corpus.values
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
# textlist = text.text_to_word_sequence(texts, split=" ")
# text.one_hot(texts, 20000, split=' ' )
tkzer = text.Tokenizer(nb_words=100000, lower=True, split=" ")

tkzer.fit_on_texts(texts)
text_seq = tkzer.texts_to_sequences(texts)

word_index = tkzer.word_index
print('Found %s unique tokens.' % len(word_index))

train_seq = text_seq[:100000]
test_seq = text_seq[100000:]

def create_ngram_set(input_list, ngram_value=2):
    
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    np.append(new_list, token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

from sklearn.cross_validation import train_test_split
# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 100000
maxlen = 500
batch_size = 128
embedding_dims = 100
nb_epoch = 20


print('Loading data...')
# X_train, X_test, y_train, y_test = train_test_split(train_seq, train_label, test_size=0.33, random_state=42)
# train_label = train.gender.values
X_train = train_seq
X_test = test_seq
# y_train = train_label

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))

if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in X_train:
        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k+start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting X_train and X_test with n-grams features
    X_train = add_ngram(X_train, token_indice, ngram_range)
    X_test = add_ngram(X_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

def build_model(cat, loss):
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))

    model.add(Dropout(0.5))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    model.add(Dropout(0.5))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(cat, activation='softmax'))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)

#gender
from sklearn.cross_validation import StratifiedKFold
from keras.utils.np_utils import to_categorical
print 'creat gender prob features'
random_seed = 2016

print('Loading data...')
train_label = train.gender.values
# X_train = train_seq
# X_test = test_seq
y_train = to_categorical(train_label)

skf = StratifiedKFold(train_label, n_folds=3, shuffle=True)

new_train = np.zeros((100000,3))
new_test = np.zeros((100000,3))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(3, 'binary_crossentropy')
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=5)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 3
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('ft_gender_',i) for i in range(3)]
stacks = np.hstack(stacks)
gender_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#age
print 'creat age prob features'
random_seed = 2016

print('Loading data...')
train_label = train.age.values
# X_train = train_seq
# X_test = test_seq
y_train = to_categorical(train_label)

skf = StratifiedKFold(train_label, n_folds=3, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(7, 'categorical_crossentropy')
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=7)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 3
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('ft_age_',i) for i in range(7)]
stacks = np.hstack(stacks)
age_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#edu
print 'creat edu prob features'
random_seed = 2016

print('Loading data...')
train_label = train.edu.values
# X_train = train_seq
# X_test = test_seq
y_train = to_categorical(train_label)

skf = StratifiedKFold(train_label, n_folds=3, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(7, 'categorical_crossentropy')
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=12)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 3
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('ft_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
lr_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
lr_prob_feat.to_csv(r'../feature/stack/ft_prob.csv', index=0)