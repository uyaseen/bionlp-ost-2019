#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
import loader
import timeit

from utils.model_utils import create_input, models_path, evaluate, eval_script, eval_temp, get_model_path, \
    get_data_strategy
from utils.utils import join_path, write_json
from loader import word_mapping, char_mapping, tag_mapping, pos_mapping, ortho_mapping, segment_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model
from evaluate import evaluate_model

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-x", "--pre_emb_1", default="",
    help="Location of 2nd pretrained embeddings"
)
optparser.add_option(
    "-X", "--pre_emb_1_dim", default="100",
    type='int', help="Dimension 2nd pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-F", "--emb_of_unk_words", default="0",
    type='int', help="Get the embeddings of OOV words from FastText?"
)
optparser.add_option(
    "-P", "--pos_dim", default="0",
    type='int', help="Include POS tags as features, number represents the dimension"
)
optparser.add_option(
    "-O", "--ortho_dim", default="0",
    type='int', help="Include orthographic shapes as features, number represents the dimension"
)
optparser.add_option(
    "-M", "--multi_task", default="0",
    type='int', help="Include entity detection as multi-tasking (only at train time)"
)
optparser.add_option(
    "-E", "--evaluation", default="pharmaco",
    help="Which Evaluation scheme to run, possible options [pharmaco, conll]"
)
optparser.add_option(
    "-V", "--ranking_loss", default="0",
    type='int', help="Also add ranking loss to the cost function"
)
optparser.add_option(
    "-J", "--language_model", default="0",
    type='int', help="Include language modelling objective as multi-tasking (only at train time)"
)

start_time = timeit.default_timer()
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['train'] = opts.train
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['emb_of_unk_words'] = opts.emb_of_unk_words == 1
parameters['pos_dim'] = opts.pos_dim
parameters['ortho_dim'] = opts.ortho_dim
parameters['multi_task'] = opts.multi_task == 1
parameters['evaluation'] = opts.evaluation
parameters['ranking_loss'] = opts.ranking_loss == 1
parameters['pre_emb_1'] = opts.pre_emb_1
parameters['pre_emb_1_dim'] = opts.pre_emb_1_dim
parameters['language_model'] = opts.language_model == 1
parameters['reload'] = opts.reload == 1

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "{}"'.format(eval_script))
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
model_path = join_path(models_path, get_model_path(parameters))
if not os.path.exists(model_path):
    os.makedirs(model_path)
write_json(parameters, join_path(model_path, 'config.json'))
# Initialize model
model = Model(parameters=parameters, model_path=model_path)
print("Model location: {}".format(model.model_path))

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train,  zeros)
dev_sentences = loader.load_sentences(opts.dev, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[1] for w in s] for s in dev_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

if parameters['pre_emb_1']:
    dico_words_train_1 = word_mapping(train_sentences, lower)[0]
    dico_words_1, word_to_id_1, id_to_word_1 = augment_with_pretrained(
        dico_words_train_1.copy(),
        parameters['pre_emb_1'],
        list(itertools.chain.from_iterable(
            [[w[1] for w in s] for s in dev_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    word_to_id_1, id_to_word_1 = {'DUMMY': 1}, {1: 'DUMMY'}

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)
pos_tags, pos_to_id, id_to_pos = pos_mapping(train_sentences + dev_sentences)
ortho_shapes, ortho_to_id, id_to_ortho = ortho_mapping(train_sentences + dev_sentences)
segment_tags, segment_to_id, id_to_segment = segment_mapping(train_sentences)

if opts.reload:
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [id_to_word, id_to_char, id_to_tag]
    ]
    pos_to_id, ortho_to_id, segment_to_id = [
        {v: k for k, v in x.items()}
        for x in [id_to_pos, id_to_ortho, id_to_segment]
    ]

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, pos_to_id, ortho_to_id, segment_to_id,
    word_to_id_1, lower=lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, pos_to_id, ortho_to_id, segment_to_id,
    word_to_id_1, lower=lower
)
print("{} / {} sentences in train / dev".format(len(train_data), len(dev_data)))


# Save the mappings to disk
print('Saving the mappings to disk...')
model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_pos, id_to_ortho, id_to_segment,
                    id_to_word_1)

# Build the model
f_train, f_eval = model.build(**parameters, word_to_id=word_to_id,
                              tag_to_id=tag_to_id)
# which strategy?
d_strategy = get_data_strategy(parameters['train'])
span_p = 'dataset/data-strategy={}/dev-token_span.pkl'.format(d_strategy)
dev_raw_files = 'dataset/raw/dev/'.format(d_strategy)
print('data_strategy::{}'.format(d_strategy))

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = 100  # number of epochs over the training set
freq_eval = 1000  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_dev_epoch = 0
count = 0
for epoch in range(n_epochs):
    epoch_costs = []
    print("Starting epoch {}...".format(epoch + 1))
    train_data_iterator = np.random.permutation(len(train_data))
    for i, index in enumerate(train_data_iterator):
        count += 1
        input = create_input(train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)
        if i % 50 == 0 and i > 0 == 0:
            print("{} / {} -- {}, cost average: {}".format(epoch + 1, n_epochs, i, np.mean(epoch_costs[-50:])))
        if count % freq_eval == 0:
            if parameters['evaluation'] == 'pharmaco':
                if not os.path.exists(join_path(model_path, 'word_layer.mat')):
                    print('saving initial model ...')
                    model.save()
                eval_score = evaluate_model(model_p=model_path, gt_p=dev_raw_files, token_span_p=span_p,
                                            f_eval=f_eval, parameters=parameters, del_results=False, return_score=True)
                dev_score = eval_score['f1']
            else:
                dev_score = evaluate(parameters, f_eval, dev_sentences,
                                     dev_data, id_to_tag)
            print("Score on dev: {}".format(dev_score))
            if dev_score > best_dev:
                best_dev = dev_score
                best_dev_epoch = epoch + 1
                print("New best score on dev.")
                print("Saving model to disk...")
                model.save()
            print("Best Score on dev: {}".format(best_dev))
            print('Best Epoch __dev__: {}'.format(best_dev_epoch))

    print("Epoch {} done. Average cost: {}".format(epoch, np.mean(epoch_costs)))
    print("Best Score on dev: {}".format(best_dev))
    print('Best Epoch __dev__: {}'.format(best_dev_epoch))
print("Best Score on dev: {}".format(best_dev))
print('Best Epoch __dev__: {}'.format(best_dev_epoch))
print("Model location: {}".format(model.model_path))
end_time = timeit.default_timer()
print('The code ran for {}m'.format(round((end_time - start_time) / 60.), 2))
