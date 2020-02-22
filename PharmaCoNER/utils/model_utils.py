import os
import re
import codecs
import numpy as np
import theano
from datetime import datetime
from utils.utils import join_path


models_path = "models/"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")
unique_phrases = set()


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter {}. Expected {}, found {}.".format(name, param_value.size, pretrained.size))
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        np.random.seed(36)
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def get_word_caps_feats_as_bow(word):
    first_word_cap = word[0] == word[0].upper()
    word_capitalized = word == word.upper()
    return [int(first_word_cap), int(word_capitalized)]


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_dico_hacky(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for item in item_list:
        dico[item] = 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def iob_ranges(tags):
    """
    IOB -> Ranges
    """
    ranges = []

    def check_if_closing_range():
        if i == len(tags) - 1 or tags[i + 1].split('-')[0] == 'O':
            ranges.append((begin, i, type))

    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'O':
            pass
        elif tag.split('-')[0] == 'B':
            begin = i
            type = tag.split('-')[1]
            check_if_closing_range()
        elif tag.split('-')[0] == 'I':
            check_if_closing_range()
    return ranges


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if parameters['pos_dim']:
        input.append(data['pos'])
    if parameters['ortho_dim']:
        input.append(data['ortho'])
    if parameters['pre_emb_1']:
        input.append(data['words_1'])
    if add_label:
        input.append(data['tags'])
    if parameters['multi_task'] and add_label:
        input.append(data['segment_tags'])
    if parameters['language_model'] and add_label:
        input.append(data['y_fwd'])
        input.append(data['y_bwd'])
    return input


def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, delete_files=True):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))[1:-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    # Remove temp files
    if delete_files:
        os.remove(output_path)
        os.remove(scores_path)

    # Confusion matrix with accuracy for each tag
    # print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
    #     "ID", "NE", "Total",
    #     *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    # )
    # for i in xrange(n_tags):
    #     print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
    #         str(i), id_to_tag[i], str(count[i].sum()),
    #         *([count[i][j] for j in xrange(n_tags)] +
    #           ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
    #     )

    # Global accuracy
    # print "%i/%i (%.5f%%)" % (
    #     count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    # )

    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])


def get_searchable_phrases(raw_sentence, p_tags):
    global unique_phrases
    searchable_tokens = ['O'] * len(p_tags)
    last_tag = 'O'
    start_chunk, end_chunk = False, False
    ph_st_idx, ph_end_idx = None, None
    last_end_idx = -1
    phrase = ''
    iter_idx = 0
    for idx in range(len(p_tags)):
        word, tag = raw_sentence[idx][1], p_tags[idx]
        if last_tag == 'O' and tag != 'O':
            start_chunk = True
            end_chunk = False
            ph_st_idx = idx
        if last_tag is not 'O' and tag == 'O':
            end_chunk = True
            start_chunk = False
            ph_end_idx = last_end_idx

        if start_chunk:
            phrase = word
            start_chunk = False
            last_tag = tag
            if iter_idx == len(p_tags) - 1:
                end_chunk = False
                start_chunk = False

        elif end_chunk:
            assert ph_st_idx is not None, 'ph_st_idx == None'
            assert ph_end_idx is not None, 'ph_end_idx == None'
            if ph_end_idx - ph_st_idx > 1:
                for i_loop in range(ph_st_idx, ph_end_idx + 1):
                    searchable_tokens[i_loop] = phrase
            else:
                searchable_tokens[ph_st_idx] = phrase
            unique_phrases.add(phrase)
            last_tag = tag
            end_chunk = False

        elif tag == 'O' and last_tag == 'O':
            # outside of the phrase or chunk
            last_tag = tag

        else:
            # middle of the chunk
            phrase = phrase + ' ' + word
            last_tag = tag
            if iter_idx == len(p_tags) - 1:
                end_chunk = False
                start_chunk = False
        iter_idx += 1
        last_end_idx = idx

    assert len(searchable_tokens) == len(p_tags), 'searchable_tokens:{}\np_tags{}\n\nphrase:{}'.format(
        searchable_tokens,
        p_tags, phrase)
    return searchable_tokens


def get_current_time():
    return datetime.now().strftime('%d-%m-%Y__%Hh.%Mm.%Ss')


def get_data_strategy(file_path):
    dirname = os.path.dirname(file_path)
    data_strategy = str(dirname.split('data-strategy=')[-1])
    assert data_strategy in ['1', '2', '3', '4', '5', '6', '7', '8']
    return data_strategy


def which_embeddings(file_path):
    emb = 'RANDOM'
    if 'fasttext' in file_path:
        emb = 'FastText'
    elif 'glove' in file_path:
        emb = 'Glove'
    elif 'docnade' in file_path:
        emb = 'DocNade'
    return emb


def get_model_path(config):
    path = ''
    if config['reload']:
        path += 'reload=True,'
    path += 'embeddings={},'.format(which_embeddings(config['pre_emb']))
    if config['pre_emb_1']:
        path += 'embeddings_1={},'.format(which_embeddings(config['pre_emb_1']))
    if config['emb_of_unk_words']:
        path += 'emb_of_unk_words=True,'
    if config['pos_dim']:
        path += 'pos_dim={},'.format(config['pos_dim'])
    if config['ortho_dim']:
        path += 'ortho_dim={},'.format(config['ortho_dim'])
    if config['lower']:
        path += 'lower=True,'
    if config['zeros']:
        path += 'lower=True,'
    if config['cap_dim']:
        path += 'cap_dim={},'.format(config['cap_dim'])
    if config['multi_task']:
        path += 'multi_task=True,'
    if not config['crf']:
        path += 'crf=False,'
    if config['ranking_loss']:
        path += 'ranking_loss=True,'
    if config['language_model']:
        path += 'language_model=True,'
    path += 'hidden_dim={},'.format(config['word_lstm_dim'])
    path += 'word_dim={},'.format(config['word_dim'])
    path += 'char_dim={},'.format(config['char_dim'])
    path += '--{}'.format(get_current_time())
    dirname = os.path.dirname(config['train'])
    data_strategy = dirname.split('/')[1]
    path = join_path(data_strategy, path)
    return path
