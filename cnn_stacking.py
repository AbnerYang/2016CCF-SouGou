import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical

train = pd.read_csv(r'../feature/raw/trainQlist.csv')
test = pd.read_csv(r'../feature/raw/testQlist.csv')
corpus = pd.concat([train.qlist, test.qlist])
corpus.index = range(corpus.shape[0])

from keras.preprocessing import sequence

nb_words = 100000
# maxlen = 500

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    corpus = pd.concat([train.qlist, test.qlist])
    corpus.index = range(corpus.shape[0])

    # positive_examples = list(open("/data/sougo/cnnExample/rt-polarity.pos").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open("/data/sougo/cnnExample/rt-polarity.neg").readlines())
    corpus = [s.strip() for s in corpus]
    # Split by words
    x_text = [s.split(" ") for s in corpus]
    return x_text

# def pad_sentences(sentences, padding_word="<PAD/>"):
#     """
#     Pads all sentences to the same length. The length is defined by the longest sentence.
#     Returns padded sentences.
#     """
#     sequence_length = max(len(x) for x in sentences)
#     padded_sentences = []
#     for i in range(len(sentences)):
#         sentence = sentences[i]
#         num_padding = sequence_length - len(sentence)
#         new_sentence = sentence + [padding_word] * num_padding
#         padded_sentences.append(new_sentence)
#     return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    word_counts['<PAD/>'] = 100000000
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    # y = np.array(labels)
    return x


def load_data():
    
    corpus = pd.concat([train.qlist, test.qlist])
    corpus.index = range(corpus.shape[0])
    
    print 'spliting words..........'
    
    corpus = [s.strip() for s in corpus]
    # Split by words
    sentences = [s.split(" ") for s in corpus]
    
    
    print 'building vovabulary................'
    vocabulary, vocabulary_inv = build_vocab(sentences)
    
#     print'Pad sequences (samples x time)'
#     sentences_padded = sequence.pad_sequences(sentences, maxlen=maxlen)
    
    x = build_input_data(sentences, vocabulary)
    
    return [x, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


sentences, vocabulary, vocabulary_inv = load_data()

from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=100, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    model_dir = '../data/word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print 'Loading existing Word2Vec model \'%s\'' % split(model_name)[-1]
    else:
        # Set values for various parameters
        num_workers = 2       # Number of threads to run in parallel
        downsampling = 1e-3   # Downsample setting for frequent words
        
        # Initialize and train the model
        print "Training Word2Vec model..."
        # sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        sentences = [s.strip().split() for s in corpus]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                            size=num_features, min_count = min_word_count, \
                            window = context, sample = downsampling)
        
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
        
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print 'Saving Word2Vec model \'%s\'' % split(model_name)[-1]
        embedding_model.save(model_name)
    
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in vocabulary_inv])]
    return embedding_weights


corpus = pd.concat([train.qlist, test.qlist])
corpus.index = range(corpus.shape[0])
w = train_word2vec(sentences, vocabulary_inv)

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D

np.random.seed(2)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

model_variation = 'CNN-non-static'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
# sequence_length = max(len(i) for i in sentences)
sequence_length = 500
maxlen = sequence_length

embedding_dim = 100          
filter_sizes = [3, 4, 5]
num_filters = 100
dropout_prob = (0.25, 0.5)
# hidden_dims = 100

# Training parameters
batch_size = 256
num_epochs = 100
val_split = 0.33

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10        # Context window size    

# Data Preparatopn
# ==================================================
#
# Load data
from keras.utils.np_utils import to_categorical
print("Loading data...")
# x, y, vocabulary, vocabulary_inv = data_helpers.load_data()
print'Pad sequences (samples x time)'
sentences_padded = sequence.pad_sequences(sentences, maxlen=maxlen)
x = sentences_padded
# X_test = sentences_padded[100000:]
# train_y = to_categorical(train.edu.values)


if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    

# # Shuffle data
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# # y_shuffled = y[shuffle_indices].argmax(axis=1)
# y_shuffled = y[shuffle_indices]
X_train = sentences_padded[:100000]
X_test = sentences_padded[100000:]
print("Vocabulary Size: {:d}".format(len(vocabulary)))   


# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
def build_model(cat, hidden_dim):
    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    if not model_variation=='CNN-static':
        model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                            weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dense(hidden_dim))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(cat))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


from sklearn.cross_validation import train_test_split

#creat cnn stacking features
#tfidf min_df=1
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

skf = StratifiedKFold(train_label, n_folds=5, shuffle=True)

new_train = np.zeros((100000,3))
new_test = np.zeros((100000,3))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(3, 100)
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=1)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('cnn_gender_',i) for i in range(3)]
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

skf = StratifiedKFold(train_label, n_folds=5, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(7, 100)
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=1)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('cnn_age_',i) for i in range(7)]
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

skf = StratifiedKFold(train_label, n_folds=5, shuffle=True)

new_train = np.zeros((100000,7))
new_test = np.zeros((100000,7))

for i,(trainid,valid) in enumerate(skf):
    print 'fold' + str(i)
    train_x = X_train[trainid]
    train_y = y_train[trainid]
    val_x = X_train[valid]
    model = build_model(7, 50)
    model.fit(train_x, train_y,
          batch_size=batch_size,
          nb_epoch=1)
    new_train[valid] = model.predict_proba(val_x)
    new_test += model.predict_proba(X_test)
    
new_test /= 5
stacks = []
stacks_name = []
stack = np.vstack([new_train,new_test])
stacks.append(stack)
stacks_name += ['%s_%d'%('cnn_edu_prob',i) for i in range(7)]
stacks = np.hstack(stacks)
edu_stacks = pd.DataFrame(data=stacks,columns=stacks_name)

#concat
cnn_prob_feat = pd.concat([age_stacks, gender_stacks, edu_stacks], axis=1)
cnn_prob_feat.to_csv(r'../feature/stack/cnn_prob.csv', index=0)