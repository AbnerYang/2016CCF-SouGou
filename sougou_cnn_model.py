# -*- encoding:UTF-8 -*-
#-- Author: TNT_000 by Abner yang
import numpy as np 
import pandas as pd 
import itertools
import time
from collections import Counter
from sklearn.cross_validation import StratifiedKFold

import mxnet as mx 

config = {
	'task':'Age'
}

def pad_sentences(sentences, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    #print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, vocabulary, trainSize):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return x[:trainSize,], x[-trainSize:,]

def load_data_and_process(config):
	# Read Raw Data 
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'start load raw data...'
	train = pd.read_csv('../feature/raw/trainQlist.csv', header = 0)
	test = pd.read_csv('../feature/raw/testQlist.csv', header = 0)
	# Read Train Label 
	trainLabel = pd.read_csv('../feature/raw/train'+config['task']+'.csv', header = 0)
	label_train = trainLabel[config['task'].lower()].values.T
	# Get Sentence
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get sentences...'
	sentences_train = [line.split(" ") for line in [sentence for sentence in train['qlist'].values]]
	sentences_test = [line.split(" ") for line in [sentence for sentence in test['qlist'].values]]
	sentences = sentences_train + sentences_test
	trainSize = len(sentences_train)
	# Pad Sentence
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'pad sentences...'	
	sentences_padded = pad_sentences(sentences)
	# Build Vocab 
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'build vocab...'
	vocabulary, vocabulary_inv = build_vocab(sentences_padded)
	# Build Input Data
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'get input data...'
	vocab_train, vocab_test = build_input_data(sentences_padded, vocabulary, trainSize)
	print vocab_train.shape, vocab_test.shape
	return vocab_train, vocab_test, label_train, vocabulary, vocabulary_inv

def make_text_cnn(sentence_size, num_embed, batch_size, vocab_size, num_label=2, filter_list=[3, 4, 5], num_filter=100, dropout=0., with_embedding=True):
	input_x = mx.sym.Variable('data') # placeholder for input
	input_y = mx.sym.Variable('softmax_label') # placeholder for output

	# embedding layer
	if not with_embedding:
		embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
		conv_input = mx.sym.Reshape(data=embed_layer, target_shape=(batch_size, 1, sentence_size, num_embed))
	else:
		conv_input = input_x

	# create convolution + (max) pooling layer for each filter operation
	pooled_outputs = []
	for i, filter_size in enumerate(filter_list):
		convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
		relui = mx.sym.Activation(data=convi, act_type='relu')
		pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1,1))
		pooled_outputs.append(pooli)

	# combine all pooled outputs
	total_filters = num_filter * len(filter_list)
	concat = mx.sym.Concat(*pooled_outputs, dim=1)
	h_pool = mx.sym.Reshape(data=concat, target_shape=(batch_size, total_filters))

	# dropout layer
	if dropout > 0.0:
		h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
	else:
		h_drop = h_pool

	# fully connected
	cls_weight = mx.sym.Variable('cls_weight')
	cls_bias = mx.sym.Variable('cls_bias')

	fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

	# softmax output
	sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

	return sm

def setup_cnn_model(ctx, batch_size, sentence_size, num_embed, vocab_size, dropout=0.5, initializer=mx.initializer.Uniform(0.1), with_embedding=True):

	cnn = make_text_cnn(sentence_size, num_embed, batch_size=batch_size,vocab_size=vocab_size, dropout=dropout, with_embedding=with_embedding)
	arg_names = cnn.list_arguments()

	input_shapes = {}
	if with_embedding:
		input_shapes['data'] = (batch_size, 1, sentence_size, num_embed)
	else:
		input_shapes['data'] = (batch_size, sentence_size)

	arg_shape, out_shape, aux_shape = cnn.infer_shape(**input_shapes)
	arg_arrays = [mx.nd.zeros(s, ctx) for s in arg_shape]
	args_grad = {}
	for shape, name in zip(arg_shape, arg_names):
		if name in ['softmax_label', 'data']: # input, output
			continue
		args_grad[name] = mx.nd.zeros(shape, ctx)

	cnn_exec = cnn.bind(ctx=ctx, args=arg_arrays, args_grad=args_grad, grad_req='add')

	param_blocks = []
	arg_dict = dict(zip(arg_names, cnn_exec.arg_arrays))
	for i, name in enumerate(arg_names):
		if name in ['softmax_label', 'data']: # input, output
			continue
		initializer(name, arg_dict[name])

		param_blocks.append( (i, arg_dict[name], args_grad[name], name) )

	out_dict = dict(zip(cnn.list_outputs(), cnn_exec.outputs))

	data = cnn_exec.arg_dict['data']
	label = cnn_exec.arg_dict['softmax_label']

	return CNNModel(cnn_exec=cnn_exec, symbol=cnn, data=data, label=label, param_blocks=param_blocks)

def train_without_pretrained_embedding(config):
	x_train, x_test, y_train, vocab, vocab_inv = load_data_and_process(config)
	vocab_size = len(vocab)
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'CV On CNN Model....'
	kfold = StratifiedKFold(y = y_train, n_folds = 3, shuffle = True, random_state = 12)

	f = 0
	predict = []
	true = []
	uid = []
	for index1, index2 in kfold:
		# split train/dev set
		x_train, x_dev = x_train[index1,], x_train[index2,]
		y_train, y_dev = y_train[index1], y_train[index2]
		print 'Train/Dev split: %d/%d' % (len(y_train), len(y_dev))
		print 'train shape:', x_train.shape
		print 'dev shape:', x_dev.shape
		print 'vocab_size', vocab_size

		batch_size = 50
		num_embed = 300
		sentence_size = x_train.shape[1]

		print 'batch size', batch_size
		print 'sentence max words', sentence_size
		print 'embedding size', num_embed

		cnn_model = setup_cnn_model(mx.gpu(0), batch_size, sentence_size, num_embed, vocab_size, dropout=0.5, with_embedding=False)
		train_cnn(cnn_model, x_train, y_train, x_dev, y_dev, batch_size)

def train_cnn(model, X_train_batch, y_train_batch, X_dev_batch, y_dev_batch, batch_size, optimizer='rmsprop', max_grad_norm=5.0, learning_rate=0.0005, epoch=200):
    m = model
    # create optimizer
    opt = mx.optimizer.create(optimizer)
    opt.lr = learning_rate

    updater = mx.optimizer.get_updater(opt)

    for iteration in range(epoch):
        tic = time.time()
        num_correct = 0
        num_total = 0
        for begin in range(0, X_train_batch.shape[0], batch_size):
            batchX = X_train_batch[begin:begin+batch_size]
            batchY = y_train_batch[begin:begin+batch_size]
            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.label[:] = batchY

            # forward
            m.cnn_exec.forward(is_train=True)

            # backward
            m.cnn_exec.backward()

            # eval on training data
            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

            # update weights
            norm = 0
            for idx, weight, grad, name in m.param_blocks:
                grad /= batch_size
                l2_norm = mx.nd.norm(grad).asscalar()
                norm += l2_norm * l2_norm

            norm = math.sqrt(norm)
            for idx, weight, grad, name in m.param_blocks:
                if norm > max_grad_norm:
                    grad *= (max_grad_norm / norm)

                updater(idx, grad, weight)

                # reset gradient to zero
                grad[:] = 0.0

        # decay learning rate
        if iteration % 50 == 0 and iteration > 0:
            opt.lr *= 0.5
            print >> logs, 'reset learning rate to %g' % opt.lr

        # end of training loop
        toc = time.time()
        train_time = toc - tic
        train_acc = num_correct * 100 / float(num_total)

        # saving checkpoint
        if (iteration + 1) % 10 == 0:
            prefix = 'cnn'
            m.symbol.save('checkpoint/%s-symbol.json' % prefix)
            save_dict = {('arg:%s' % k) :v  for k, v in m.cnn_exec.arg_dict.items()}
            save_dict.update({('aux:%s' % k) : v for k, v in m.cnn_exec.aux_dict.items()})
            param_name = 'checkpoint/%s-%04d.params' % (prefix, iteration)
            mx.nd.save(param_name, save_dict)
            print >> logs, 'Saved checkpoint to %s' % param_name


        # evaluate on dev set
        num_correct = 0
        num_total = 0
        for begin in range(0, X_dev_batch.shape[0], batch_size):
            batchX = X_dev_batch[begin:begin+batch_size]
            batchY = y_dev_batch[begin:begin+batch_size]

            if batchX.shape[0] != batch_size:
                continue

            m.data[:] = batchX
            m.cnn_exec.forward(is_train=False)

            num_correct += sum(batchY == np.argmax(m.cnn_exec.outputs[0].asnumpy(), axis=1))
            num_total += len(batchY)

        dev_acc = num_correct * 100 / float(num_total)
        print >> logs, 'Iter [%d] Train: Time: %.3fs, Training Accuracy: %.3f Dev Accuracy thus far: %.3f' % (iteration, train_time, train_acc, dev_acc)

	


if __name__ == '__main__':
	train_without_pretrained_embedding(config)


