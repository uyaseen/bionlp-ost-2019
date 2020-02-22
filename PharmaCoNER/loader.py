import os
import re
import codecs
from utils.model_utils import create_dico, create_mapping, zero_digits, iob2, iob_iobes
from utils.model_utils import get_word_caps_feats_as_bow


def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence {}:\n{}'.format(i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[1].lower() if lower else x[1] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    dico['<PADDING>'] = 500000
    word_to_id, id_to_word = create_mapping(dico)
    print('Found {} unique words ({} in total)'.format(len(dico), sum(len(x) for x in words)))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[1] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found {} unique characters".format(len(dico)))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences, idx=-1):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[idx] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found {} unique named entity tags".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def pos_mapping(sentences):
    """
    Create a dictionary and a mapping of POS tags, sorted by frequency.
    """
    tags = [[word[2] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico['<UNK>'] = 10000000
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found {} unique POS tags".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def ortho_mapping(sentences):
    """
    Create a dictionary and a mapping of ortho tags, sorted by frequency.
    """
    tags = [[word[3] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico['<UNK>'] = 10000000
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found {} unique ortho tags".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def segment_mapping(sentences, only_i_o_tag=True):
    """
    Create a dictionary and a mapping of ner segmentation tags, sorted by frequency.
    """
    if only_i_o_tag:
        tags = [[word[4].replace('B-', 'I-') for word in s] for s in sentences]
    else:
        tags = [[word[4] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found {} unique segmentation tags".format(len(dico)))
    return dico, tag_to_id, id_to_tag


def is_punctuation(word):
    for w in word:
        if w.isalnum():
            return False
    return True


def is_digit(word):
    try:
        _ = float(word.replace(' ', ''))
        return True
    except ValueError:
        return False


def cap_feature(s):
    """
    Capitalization feature:
    0 = digit
    1 = punctuation
    2 = word in lower case
    3 = whole word in caps
    4 = one capital (not first letter)
    """
    # if is_digit(s):
    if s.isdigit():
        return 0
    if is_punctuation(s):
        return 1
    # so word is at least alpha-numeric
    if s.lower() == s:
        return 2
    elif s.upper() == s:
        return 3
    elif s[0].upper() == s[0]:
        return 4
    else:
        return 5


def prepare_sentence(input_dict, word_to_id, char_to_id, pos_to_id, ortho_to_id, segment_to_id,
                     word_to_id_1, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x

    str_words = input_dict['words']
    pos_tags = input_dict['pos']
    ortho_tags = input_dict['ortho']
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    y_fwd = words[1:] + words[:1]
    y_bwd = words[::-1][1:] + words[::-1][:1]
    words_1 = [word_to_id_1[f(w) if f(w) in word_to_id_1 else '<UNK>']
               for w in str_words] if len(word_to_id_1) > 1 else None
    chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    pos = [pos_to_id[p if p in pos_to_id else '<UNK>'] for p in pos_tags]
    ortho = [ortho_to_id[o if o in ortho_to_id else '<UNK>'] for o in ortho_tags]
    segment = [segment_to_id['O'] for _ in ortho_tags]

    return {
        'y_fwd': y_fwd,
        'y_bwd': y_bwd,
        'str_words': str_words,
        'words': words,
        'words_1': words_1,
        'chars': chars,
        'caps': caps,
        'pos': pos,
        'ortho': ortho,
        'segment_tags': segment
    }


def segmentation_tag(tag):
    segmented = None
    if tag == 'O':
        segmented = 'O'
    elif tag.startswith('B-'):
        segmented = 'B-SEGMENT'
    elif tag.startswit('I-'):
        segmented = 'I-SEGMENT'
    return segmented


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, pos_to_id, ortho_to_id,
                    segment_to_id, word_to_id_1, lower=False, only_i_o_tags=True, tag_idx=-1):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[1] for w in s]
        pos_tags = [w[2] for w in s]
        ortho_tags = [w[3] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        y_fwd = words[1:] + words[:1]
        y_bwd = words[::-1][1:] + words[::-1][:1]
        words_1 = [word_to_id_1[f(w) if f(w) in word_to_id_1 else '<UNK>']
                   for w in str_words] if len(word_to_id_1) > 1 else None
        # Skip characters that are not in the training set
        chars = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        w_caps_bow = [get_word_caps_feats_as_bow(w) for w in str_words]
        pos = [pos_to_id[p if p in pos_to_id else '<UNK>'] for p in pos_tags]
        ortho = [ortho_to_id[o if o in ortho_to_id else '<UNK>'] for o in ortho_tags]
        if only_i_o_tags:
            segment_tags = [segment_to_id[w[4].replace('B-', 'I-')] for w in s]
        else:
            segment_tags = [segment_to_id[w[4]] for w in s]
        tags = [tag_to_id[w[tag_idx]] for w in s]

        data.append({
            'y_fwd': y_fwd,
            'y_bwd': y_bwd,
            'str_words': str_words,
            'words': words,
            'words_1': words_1,
            'chars': chars,
            'caps': caps,
            'tags': tags,
            'word_caps_bow': w_caps_bow,
            'pos': pos,
            'ortho': ortho,
            'segment_tags': segment_tags
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pre-trained embeddings from {}...'.format(ext_emb_path))
    assert os.path.isfile(ext_emb_path), '{} not found'.format(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word
