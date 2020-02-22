import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs

from theano.ifelse import ifelse
from fasttext import load_model
from utils.utils import read_pickle, write_pickle
from utils.model_utils import shared, set_values
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward, DummyLayer
from optimization import Optimization


class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, model_path=None, to_save=True):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        self.parameters_path = os.path.join(model_path, 'parameters.pkl')
        self.mappings_path = os.path.join(model_path, 'mappings.pkl')
        self.model_path = model_path
        if parameters:
            self.parameters = parameters
            # Create directory for the model if it does not exist
            if not os.path.exists(model_path):
                print('current directory: {}'.format(os.getcwd()))
                print('models_path: {}'.format(model_path))
                os.makedirs(model_path)
            if to_save:
                # Save the parameters to disk
                write_pickle(parameters, self.parameters_path)
        else:
            self.parameters = read_pickle(self.parameters_path)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag, id_to_pos, id_to_ortho, id_to_segment, id_to_word_1):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        self.id_to_pos = id_to_pos
        self.id_to_ortho = id_to_ortho
        self.id_to_segment = id_to_segment
        self.id_to_word_1 = id_to_word_1
        mappings = {
            'id_to_word': self.id_to_word,
            'id_to_char': self.id_to_char,
            'id_to_tag': self.id_to_tag,
            'id_to_pos': self.id_to_pos,
            'id_to_ortho': self.id_to_ortho,
            'id_to_segment': self.id_to_segment,
            'id_to_word_1': self.id_to_word_1
        }
        write_pickle(mappings, self.mappings_path)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        print('reload mappings ....')
        mappings = read_pickle(self.mappings_path)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']
        self.id_to_pos = mappings['id_to_pos']
        self.id_to_ortho = mappings['id_to_ortho']
        self.id_to_segment = mappings['id_to_segment']
        self.id_to_word_1 = mappings['id_to_word_1']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "{}"!'.format(param.name))
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "{}.mat".format(name))
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self, to_ignore='SAL_WEIGHTS'):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "{}.mat".format(name))
            if to_ignore in param_path:
                continue
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def reload_from(self, ref_model_path=None, to_ignore='SAL_WEIGHTS'):
        """
        Load components values from disk.
        """
        assert ref_model_path is not None
        for name, param in self.components.items():
            param_path = os.path.join(ref_model_path, "{}.mat".format(name))
            if to_ignore in param_path:
                continue
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              training=True,
              word_to_id=None,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 6

        if self.parameters['pos_dim']:
            n_pos = len(self.id_to_pos)
        if self.parameters['ortho_dim']:
            n_ortho = len(self.id_to_ortho)
        if self.parameters['multi_task']:
            n_segment_tags = len(self.id_to_segment)
        if self.parameters['pre_emb_1_dim']:
            n_words_1 = len(self.id_to_word_1)

            # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')
        if self.parameters['pos_dim'] :
            pos_ids = T.ivector(name='pos_ids')
        if self.parameters['ortho_dim']:
            ortho_ids = T.ivector(name='ortho_ids')
        if self.parameters['multi_task']:
            segment_tags_ids = T.ivector(name='segment_tags_ids')
        if self.parameters['pre_emb_1_dim']:
            word_ids_1 = T.ivector(name='doc_ids_dn')
        if self.parameters['language_model']:
            y_fwd_ids = T.ivector(name='y_fwd_ids')
            y_bwd_ids = T.ivector(name='y_bwd_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        #
        # Word inputs
        #
        if word_dim:
            input_dim += word_dim
            print('word_dim: {}'.format(word_dim))
            word_layer = EmbeddingLayer(n_words, word_dim,
                                        name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training and not self.parameters['reload']:
                new_weights = word_layer.embeddings.get_value()
                print('Loading pretrained embeddings from {}...'.format(pre_emb))
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print('WARNING: {} invalid lines'.format(emb_invalid))
                c_found = 0
                c_lower = 0
                c_zeros = 0
                oov_words = 0
                if self.parameters['emb_of_unk_words']:
                    # TODO
                    # add path as a parameter
                    fast_text_model_p = '/home/ubuntu/usama_ws/resources/Spanish-Corporas/embeddings/fasttext/' \
                                        'fasttext-100d.bin'
                    ft_model = load_model(fast_text_model_p)
                # Lookup table initialization
                for i in range(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                    else:
                        if self.parameters['emb_of_unk_words']:
                            new_weights[i] = ft_model.get_word_vector(word)
                        oov_words += 1

                # set row corresponding to padding token to 0
                new_weights[word_to_id['<PADDING>']] = np.zeros(word_dim)

                word_layer.embeddings.set_value(new_weights)
                print('Loaded {} pretrained embeddings.'.format(len(pretrained)))
                print('{} / {} ({} percent) words have been initialized with '
                      'pretrained embeddings.'.format(
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words
                ))
                print('{} found directly, {} after lowercasing, '
                      '{} after lowercasing + zero.'.format(
                    c_found, c_lower, c_zeros
                ))
                print('oov words count: {}'.format(oov_words))

        #
        # Word inputs
        #
        if self.parameters['pre_emb_1']:
            print('pre_emb_1_dim: {}'.format(self.parameters['pre_emb_1_dim']))
            input_dim += self.parameters['pre_emb_1_dim']
            word_layer_1 = EmbeddingLayer(n_words_1, word_dim,
                                          name='word_layer_1')
            word_input_1 = word_layer_1.link(word_ids_1)
            inputs.append(word_input_1)
            if training and not self.parameters['reload']:
                # Initialize with pretrained embeddings
                new_weights_1 = word_layer_1.embeddings.get_value()
                print('Loading pretrained embeddings from {}...'.format(self.parameters['pre_emb_1']))
                pretrained_1 = {}
                emb_invalid_1 = 0
                for i, line in enumerate(codecs.open(self.parameters['pre_emb_1'], 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == self.parameters['pre_emb_1_dim'] + 1:
                        pretrained_1[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid_1 += 1
                if emb_invalid_1 > 0:
                    print('WARNING: {} invalid lines'.format(emb_invalid_1))
                c_found = 0
                c_lower = 0
                c_zeros = 0
                oov_words = 0
                # Lookup table initialization
                for i in range(n_words_1):
                    word_1 = self.id_to_word_1[i]
                    if word_1 in pretrained_1:
                        new_weights_1[i] = pretrained_1[word_1]
                        c_found += 1
                    elif word_1.lower() in pretrained_1:
                        new_weights_1[i] = pretrained_1[word_1.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word_1.lower()) in pretrained_1:
                        new_weights_1[i] = pretrained_1[
                            re.sub('\d', '0', word_1.lower())
                        ]
                        c_zeros += 1
                    else:
                        oov_words += 1

                word_layer_1.embeddings.set_value(new_weights_1)
                print('Loaded {} pretrained embeddings.'.format(len(pretrained_1)))
                print('{} / {} ({} percent) words have been initialized with '
                      'pretrained embeddings.'.format(
                    c_found + c_lower + c_zeros, n_words,
                    100. * (c_found + c_lower + c_zeros) / n_words
                ))
                print('{} found directly, {} after lowercasing, '
                      '{} after lowercasing + zero.'.format(
                    c_found, c_lower, c_zeros
                ))
                print('oov words count: {}'.format(oov_words))

        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))
        if self.parameters['pos_dim']:
            input_dim += self.parameters['pos_dim']
            pos_layer = EmbeddingLayer(n_pos, self.parameters['pos_dim'], name='pos_layer')
            inputs.append(pos_layer.link(pos_ids))
            # zeroing the '<UNK>' pos tag row
            # loading reverse mappings
            pos_to_id = {y: x for x, y in self.id_to_pos.items()}
            unk_idx = pos_to_id['<UNK>']
            _pos_wts = pos_layer.embeddings.get_value()
            _pos_wts[unk_idx] = [0.] * self.parameters['pos_dim']
            pos_layer.embeddings.set_value(_pos_wts)
        if self.parameters['ortho_dim']:
            input_dim += self.parameters['ortho_dim']
            ortho_layer = EmbeddingLayer(n_ortho, self.parameters['ortho_dim'], name='ortho_layer')
            inputs.append(ortho_layer.link(ortho_ids))
            ortho_to_id = {y: x for x, y in self.id_to_ortho.items()}
            unk_idx = ortho_to_id['<UNK>']
            _pos_wts = ortho_layer.embeddings.get_value()
            _pos_wts[unk_idx] = [0.] * self.parameters['ortho_dim']
            ortho_layer.embeddings.set_value(_pos_wts)

        print('input_dim: {}'.format(input_dim))
        # Prepare final input
        inputs = T.concatenate(inputs, axis=1) if len(inputs) != 1 else inputs[0]
        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            n_h = 2 * word_lstm_dim
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(n_h, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)
        if self.parameters['multi_task']:
            # Sentence to Named Entity Segmentation tags - Score
            segment_layer = HiddenLayer(word_lstm_dim, n_segment_tags, name='segment_layer',
                                        activation=(None if crf else 'softmax'))
            segment_tags_scores = segment_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
            if self.parameters['multi_task']:
                cost_segment = T.nnet.categorical_crossentropy(segment_tags_scores, segment_tags_ids).mean()
                cost += cost_segment
        # CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )

            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            cost = - (real_path_score - all_paths_scores)

            if self.parameters['multi_task']:
                segment_transitions = shared((n_segment_tags + 2, n_segment_tags + 2),
                                             'segment_transitions')

                seg_small = -1000
                seg_b_s = np.array([[seg_small] * n_segment_tags + [0, seg_small]]).astype(np.float32)
                seg_e_s = np.array([[seg_small] * n_segment_tags + [seg_small, 0]]).astype(np.float32)
                segment_observations = T.concatenate(
                    [segment_tags_scores, seg_small * T.ones((s_len, 2))],
                    axis=1
                )
                segment_observations = T.concatenate(
                    [seg_b_s, segment_observations, seg_e_s],
                    axis=0
                )

                # Score from tags
                seg_real_path_score = segment_tags_scores[T.arange(s_len), segment_tags_ids].sum()

                # Score from transitions
                seg_b_id = theano.shared(value=np.array([n_segment_tags], dtype=np.int32))
                seg_e_id = theano.shared(value=np.array([n_segment_tags + 1], dtype=np.int32))
                seg_padded_tags_ids = T.concatenate([seg_b_id, segment_tags_ids, seg_e_id], axis=0)
                seg_real_path_score += segment_transitions[
                    seg_padded_tags_ids[T.arange(s_len + 1)],
                    seg_padded_tags_ids[T.arange(s_len + 1) + 1]
                ].sum()

                seg_all_paths_scores = forward(segment_observations, segment_transitions)
                cost_segment = - (seg_real_path_score - seg_all_paths_scores)
                cost += cost_segment

        if training and self.parameters['ranking_loss']:
            def recurrence(x_t, y_t):
                token_prob_pos = x_t[y_t]
                arg_max_1 = T.argmax(x_t)
                arg_max_2 = T.argsort(-x_t)[1]
                token_prob_neg = ifelse(T.eq(y_t, arg_max_1), x_t[arg_max_2], x_t[arg_max_1])
                cost_t = T.max([0, 1.0 - token_prob_pos + token_prob_neg])
                return cost_t

            cost_r, _ = theano.scan(recurrence,
                                    sequences=[tags_scores, tag_ids])
            cum_cost = T.sum(cost_r)
            cost += cum_cost

        if self.parameters['language_model']:
            lm_fwd_layer = HiddenLayer(word_lstm_dim, n_words,
                                       name='lm_fwd_layer', activation='softmax')
            lm_fwd_scores = lm_fwd_layer.link(final_output)
            lm_fwd_cost = T.nnet.categorical_crossentropy(lm_fwd_scores, y_fwd_ids).mean()
            lm_bwd_layer = HiddenLayer(word_lstm_dim, n_words,
                                       name='lm_bwd_layer', activation='softmax')
            lm_bwd_scores = lm_bwd_layer.link(final_output)
            lm_bwd_cost = T.nnet.categorical_crossentropy(lm_bwd_scores, y_bwd_ids).mean()
            cost_lm = lm_fwd_cost + lm_bwd_cost
            cost += cost_lm

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if self.parameters['pre_emb_1']:
            self.add_component(word_layer_1)
            params.extend(word_layer_1.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        if self.parameters['pos_dim']:
            self.add_component(pos_layer)
            params.extend(pos_layer.params)
        if self.parameters['ortho_dim']:
            self.add_component(ortho_layer)
            params.extend(ortho_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if self.parameters['multi_task']:
            self.add_component(segment_layer)
            params.extend(segment_layer.params)
        if crf:
            self.add_component(transitions)
            params.append(transitions)
            if self.parameters['multi_task']:
                self.add_component(segment_transitions)
                params.append(segment_transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)
        if self.parameters['language_model']:
            self.add_component(lm_fwd_layer)
            params.extend(lm_fwd_layer.params)
            self.add_component(lm_bwd_layer)
            params.extend(lm_bwd_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        if self.parameters['pos_dim']:
            eval_inputs.append(pos_ids)
        if self.parameters['ortho_dim']:
            eval_inputs.append(ortho_ids)
        if self.parameters['pre_emb_1']:
            eval_inputs.append(word_ids_1)
        train_inputs = eval_inputs + [tag_ids]
        if self.parameters['multi_task']:
            train_inputs += [segment_tags_ids]
        if self.parameters['language_model']:
            train_inputs.append(y_fwd_ids)
            train_inputs.append(y_bwd_ids)

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print('Compiling...')
        if training:
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {}),
                allow_input_downcast=True
            )
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                allow_input_downcast=True
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                allow_input_downcast=True
            )

        return f_train, f_eval
