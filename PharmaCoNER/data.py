# Parse raw data of PharmaCoNER and convert it into BIO format
from Med_Tagger import Med_Tagger
from utils.utils import get_files, get_filename, join_path, file_exists, read_pickle, create_directory, \
    get_parent_directory, write_pickle

import codecs
import timeit


def tokenize(text, tagger, return_pos=True):
    '''
    :param text: text string to tokenize/pos-tag
    :param tagger: pointer to the docker image
    :param return_pos: also return POS tags for text
    :return: sentences_tokens, sentences_pos
    '''

    # med tagger for some reasons quite often combines some tokens with '_', an ugly hack to cater for that:
    # i. replace every underscore with fixed token like 123456
    # ii. after the tokenized text is returned, replace 123456 with '_'
    # iii. split the token containing underscore to reverse the effect of combining words
    text_modified = text.replace('_', '12345678')
    parsed_text = tagger.parse(text_modified)
    if len(parsed_text) == 0:
        if return_pos:
            return None, None
        return None
    if type(parsed_text[0]) == tuple:
        parsed_text_t = []
        parsed_text_t.append(parsed_text)
        parsed_text = parsed_text_t
    sentences_tokens, sentences_pos = [], []
    for sentence in parsed_text:
        tokens_t = [w_info[0] for w_info in sentence]
        pos_tags_t = [w_info[2] for w_info in sentence]
        tokens_clean, pos_clean = [], []
        for token, pos in zip(tokens_t, pos_tags_t):
            # med-tagger has merged two tokens into one via an underscore
            if '_' in token:
                token_sp = token.split('_')
                for tkn in token_sp:
                    tokens_clean.append(tkn)
                    # same pos tags to the merged tokens
                    pos_clean.append(pos)
            else:
                tokens_clean.append(token)
                pos_clean.append(pos)
        # replace our introduced string 123456 with underscore to make sure we return the original text
        for idx in range(len(tokens_clean)):
            tokens_clean[idx] = tokens_clean[idx].replace('12345678', '_')
        assert len(tokens_clean) == len(pos_clean)
        sentences_tokens.append(tokens_clean)
        sentences_pos.append(pos_clean)

    if return_pos:
        return sentences_tokens, sentences_pos
    return sentences_tokens


def get_ortho_feature(word):
    ortho = ''
    for letter in word:
        if letter.isalpha():
            if letter.isupper():
                ortho = ortho + 'C'
            else:
                ortho = ortho + 'c'
        elif letter.isdigit():
            ortho = ortho + 'n'
        else:
            ortho = ortho + 'p'
    return ortho


def recur(text):
    markers = ['\xa0', '\n', '\x85', ' ']
    for m in markers:
        if text.startswith(m):
            return True
    return False


def adjust_span(text, start, end, update_end=True):
    while recur(text[start: end]):
        if text[start: end].startswith('\xa0') or text[start: end].startswith('\n') or \
                text[start: end].startswith('\x85') or text[start: end].startswith(' '):
            start += 1
            if update_end:
                end += 1
    return start, end


def get_real_token_span(directory):
    '''
    :param directory: path of raw text files
    :return: documents :: dictionary --> key: document name, values = [[[{'word': XX, 'start': X, 'end': X}]]]
    '''
    files = get_files(directory, ext='txt')
    documents = {}
    med_tagger = Med_Tagger()  # Starts a docker image in background
    file_counter = 0
    print('get_real_token_span::{} files to process'.format(len(files)))
    for file in files:
        file_counter += 1
        if file_counter % 100 == 0:
            print('.')
        with codecs.open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens_space_offsets = []
        text_space_sp = text.split(' ')
        off_set = 0
        for token in text_space_sp:
            token_offset = {'token': token,
                            'start': off_set,
                            'end': off_set + len(token)}
            tokens_space_offsets.append(token_offset)
            off_set += len(token)
            off_set += 1
        # sanity check if captured token indexes are correct across all the tokens
        for t_offset in tokens_space_offsets:
            token = t_offset['token']
            off_set_st = t_offset['start']
            off_set_end = t_offset['end']
            assert token == text[off_set_st: off_set_end]
        tokens_space_offsets_ptr = 0
        last_inside_token_end_idx = None
        sentences_in_doc = []
        for line in codecs.open(file, 'r', encoding='utf-8'):
            if line.strip():
                sentences_tokenized, sentences_pos = tokenize(line.strip(), med_tagger, return_pos=True)
                assert sentences_tokenized is not None and sentences_pos is not None
                for sentence_tokenized, sentence_pos in zip(sentences_tokenized, sentences_pos):
                    words_in_sentence = []
                    # sentence_tokenized = web_tokenizer(sent)
                    for word, pos in zip(sentence_tokenized, sentence_pos):
                        token = tokens_space_offsets[tokens_space_offsets_ptr]['token']
                        start = tokens_space_offsets[tokens_space_offsets_ptr]['start']
                        end = tokens_space_offsets[tokens_space_offsets_ptr]['end']
                        if word == token:
                            tokens_space_offsets_ptr += 1
                            last_inside_token_end_idx = None
                        elif word in token:
                            if not last_inside_token_end_idx:
                                # remove \n from the start of text
                                adjust_span(text, start, end, update_end=False)
                                end = start + len(word)
                            else:
                                start = last_inside_token_end_idx
                                end = start + len(word)
                                start, end = adjust_span(text, start, end)

                            start, end = adjust_span(text, start, end)
                            end_of_token_space = tokens_space_offsets[tokens_space_offsets_ptr]['end']
                            # is it the end of current word, if yes increment the pointer
                            if end == end_of_token_space:
                                tokens_space_offsets_ptr += 1
                                last_inside_token_end_idx = None
                            else:
                                last_inside_token_end_idx = end
                            # part of ugly checks (pharmaco::->test)
                            if text[end:end_of_token_space] == '\x85':
                                tokens_space_offsets_ptr += 1
                                last_inside_token_end_idx = None
                        # final sanity test
                        assert word == text[start: end], 'word={} != text={}\ntokens_space_offsets_ptr:{}' \
                                                         '\nsent_tokenized: {}\ndocument: {}'.\
                            format(word, text[start: end], tokens_space_offsets_ptr,
                                   sentence_tokenized, get_filename(file))
                        words_dictio = {'word': word, 'start': start, 'end': end, 'pos': pos}
                        words_in_sentence.append(words_dictio)

                    sentences_in_doc.append(words_in_sentence)

        f_name = get_filename(file)
        documents[f_name] = sentences_in_doc
    # clean-up
    del med_tagger
    return documents


def parse_annotation_file(path):
    entities = []
    for line in codecs.open(path, 'r', encoding='utf-8'):
        if line.strip():
            if not line.startswith('T'):
                continue
            _, label_and_span, text = line.strip().split('\t')
            label, start, end = label_and_span.split()
            if type(text) == list:
                text = ' '.join(text)
            entities.append({'label': label,
                             'start': int(start),
                             'end': int(end),
                             'text': text})
    return entities


def write_bio(path, documents_tokens, documents_tags, documents_pos,
              documents_ortho, documents_segment, documents_fname):
    assert len(documents_tokens) == len(documents_tags) == len(documents_pos) == len(documents_ortho) == \
        len(documents_segment) == len(documents_fname)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for idx in range(len(documents_tokens)):
            document_tokens, document_tags, document_pos, document_ortho, document_segment, document_fname = \
                documents_tokens[idx], documents_tags[idx], documents_pos[idx], documents_ortho[idx], \
                documents_segment[idx], documents_fname[idx]
            for s_idx in range(len(document_tokens)):
                sentence_tokens, sentence_tags, sentence_pos, sentence_ortho, sentence_segment, sentence_fname = \
                    document_tokens[s_idx], document_tags[s_idx], document_pos[s_idx], document_ortho[s_idx], \
                    document_segment[s_idx], document_fname[s_idx]
                for w_idx in range(len(sentence_tokens)):
                    word, tag, pos, ortho, segment, fname = \
                        sentence_tokens[w_idx], sentence_tags[w_idx], sentence_pos[w_idx], sentence_ortho[w_idx], \
                        sentence_segment[w_idx], sentence_fname[w_idx]
                    f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fname, word, pos, ortho, segment, tag))
                f.write('\n')
    print('{} created'.format(path))


def write_bio_test(path, documents_tokens, documents_pos, documents_ortho, documents_fname, sentence_level=True):
    with codecs.open(path, 'w', encoding='utf-8') as f:
        for document_tokens, document_pos, document_ortho, document_fname in \
                zip(documents_tokens, documents_pos, documents_ortho, documents_fname):
            for sentence_tokens, sentence_pos, sentence_ortho, sentence_fname in \
                    zip(document_tokens, document_pos, document_ortho, document_fname):
                for word, pos, ortho, fname in zip(sentence_tokens, sentence_pos, sentence_ortho, sentence_fname):
                    f.write('{}\t{}\t{}\t{}\n'.format(fname, word, pos, ortho))
                if sentence_level:
                    f.write('\n')
            if not sentence_level:
                f.write('\n')
    print('{} created'.format(path))


def is_token_an_entity(token_dict, entity_dict):
    text, start, end = token_dict['word'], int(token_dict['start']), int(token_dict['end'])
    for entity in entity_dict:
        entity_text = entity['text']
        entity_start = int(entity['start'])
        entity_end = int(entity['end'])
        entity_label = entity['label']
        if start >= entity_start and end <= entity_end:
            assert text in entity_text, 'text={}, entity_text={}'.format(text, entity_text)
            return True, entity_label
    return False, 'O'


def parse(data_path, w_path, doc_token_span_w_path=None, ann_file_ext='ann',
          append_i_tag=True):
    create_directory(get_parent_directory(w_path))
    if not file_exists(doc_token_span_w_path):
        print('{} not found, computing doc-level-span information dictionary'.format(doc_token_span_w_path))
        documents_spans = get_real_token_span(data_path)
        # keep a copy of token spans to avoid re-computing it during training etc.,
        write_pickle(documents_spans, doc_token_span_w_path)
        print('{} created'.format(doc_token_span_w_path))
    else:
        documents_spans = read_pickle(doc_token_span_w_path)
    txt_files = get_files(data_path, ext='txt')
    documents_tokens = []
    documents_tags = []
    documents_pos = []
    documents_ortho = []
    documents_segment = []
    documents_fname = []
    for txt_path in txt_files:
        document_tokens = []
        document_tags = []
        document_pos = []
        document_ortho = []
        document_segment = []
        document_fname = []
        att_path = join_path(data_path, '{}.{}'.format(get_filename(txt_path), ann_file_ext))
        entities_dict = parse_annotation_file(att_path)
        f_name = get_filename(txt_path)
        sentences = documents_spans[f_name]
        for sentence in sentences:
            sentence_tokens = []
            sentence_tags = []
            sentence_pos = []
            sentence_ortho = []
            sentence_segment = []
            sentence_fname = []
            for word_dictio in sentence:
                _, tag = is_token_an_entity(word_dictio, entities_dict)
                if append_i_tag:
                    if tag != 'O':
                        tag = 'I-{}'.format(tag)
                segment = 'O' if tag == 'O' else 'I-SEGMENT'
                sentence_tokens.append(word_dictio['word'])
                sentence_tags.append(tag)
                sentence_pos.append(word_dictio['pos'])
                sentence_ortho.append(get_ortho_feature(word_dictio['word']))
                sentence_segment.append(segment)
                sentence_fname.append(f_name)
            document_tokens.append(sentence_tokens)
            document_tags.append(sentence_tags)
            document_pos.append(sentence_pos)
            document_ortho.append(sentence_ortho)
            document_segment.append(sentence_segment)
            document_fname.append(sentence_fname)
        documents_tokens.append(document_tokens)
        documents_tags.append(document_tags)
        documents_pos.append(document_pos)
        documents_ortho.append(document_ortho)
        documents_segment.append(document_segment)
        documents_fname.append(document_fname)
    write_bio(w_path, documents_tokens, documents_tags, documents_pos,
              documents_ortho, documents_segment, documents_fname)


# exactly the same as above parse function but instead it just takes a list of files
# (needed when creating multiple splits)
def parse_from_list(txt_files, w_path, doc_token_span_w_path, train_data_path, dev_data_path,
                    ann_file_ext='ann', append_i_tag=True):
    assert doc_token_span_w_path is not None
    documents_spans = read_pickle(doc_token_span_w_path)
    documents_tokens = []
    documents_tags = []
    documents_pos = []
    documents_ortho = []
    documents_segment = []
    documents_fname = []
    # 'txt_path' is a misnomer, instead of path it's just a file name without the extension
    for txt_path in txt_files:
        document_tokens = []
        document_tags = []
        document_pos = []
        document_ortho = []
        document_segment = []
        document_fname = []
        att_path = join_path(train_data_path, '{}.{}'.format(txt_path, ann_file_ext))
        if not file_exists(att_path):
            att_path = join_path(dev_data_path, '{}.{}'.format(txt_path, ann_file_ext))
        entities_dict = parse_annotation_file(att_path)
        f_name = txt_path
        sentences = documents_spans[f_name]
        for sentence in sentences:
            sentence_tokens = []
            sentence_tags = []
            sentence_pos = []
            sentence_ortho = []
            sentence_segment = []
            sentence_fname = []
            for word_dictio in sentence:
                _, tag = is_token_an_entity(word_dictio, entities_dict)
                if append_i_tag:
                    if tag != 'O':
                        tag = 'I-{}'.format(tag)
                segment = 'O' if tag == 'O' else 'I-SEGMENT'
                sentence_tokens.append(word_dictio['word'])
                sentence_tags.append(tag)
                sentence_pos.append(word_dictio['pos'])
                sentence_ortho.append(get_ortho_feature(word_dictio['word']))
                sentence_segment.append(segment)
                sentence_fname.append(f_name)
            document_tokens.append(sentence_tokens)
            document_tags.append(sentence_tags)
            document_pos.append(sentence_pos)
            document_ortho.append(sentence_ortho)
            document_segment.append(sentence_segment)
            document_fname.append(sentence_fname)
        documents_tokens.append(document_tokens)
        documents_tags.append(document_tags)
        documents_pos.append(document_pos)
        documents_ortho.append(document_ortho)
        documents_segment.append(document_segment)
        documents_fname.append(document_fname)
    write_bio(w_path, documents_tokens, documents_tags, documents_pos,
              documents_ortho, documents_segment, documents_fname)


def parse_test(data_path, w_path, doc_w_path=None, doc_token_span_w_path=None):
    if doc_token_span_w_path and not file_exists(doc_token_span_w_path):
        print('{} not found, computing doc-level-span information dictionary'.format(doc_token_span_w_path))
        documents_spans = get_real_token_span(data_path)
        # keep a copy of token spans to avoid re-computing it during training etc.,
        write_pickle(documents_spans, doc_token_span_w_path)
        print('{} created'.format(doc_token_span_w_path))
    else:
        documents_spans = read_pickle(doc_token_span_w_path)
    txt_files = get_files(data_path, ext='txt')
    documents_tokens = []
    documents_pos = []
    documents_ortho = []
    documents_fname = []
    for txt_path in txt_files:
        document_tokens = []
        document_pos = []
        document_ortho = []
        document_fname = []
        f_name = get_filename(txt_path)
        sentences = documents_spans[f_name]
        for sentence in sentences:
            sentence_tokens = []
            sentence_pos = []
            sentence_ortho = []
            sentence_fname = []
            for word_dictio in sentence:
                sentence_tokens.append(word_dictio['word'])
                sentence_pos.append(word_dictio['pos'])
                sentence_ortho.append(get_ortho_feature(word_dictio['word']))
                sentence_fname.append(f_name)
            document_tokens.append(sentence_tokens)
            document_pos.append(sentence_pos)
            document_ortho.append(sentence_ortho)
            document_fname.append(sentence_fname)
        documents_tokens.append(document_tokens)
        documents_pos.append(document_pos)
        documents_ortho.append(document_ortho)
        documents_fname.append(document_fname)
    write_bio_test(w_path, documents_tokens, documents_pos,
                   documents_ortho, documents_fname, sentence_level=True)
    if doc_w_path:
        write_bio_test(doc_w_path, documents_tokens, documents_pos,
                       documents_ortho, documents_fname, sentence_level=False)


def create_data_splits(base_path, span_path, data_path):
    train_tokens_span = read_pickle(join_path(span_path, 'train-token_span.pkl'))
    dev_tokens_span = read_pickle(join_path(span_path, 'dev-token_span.pkl'))
    split_2_tr = {}
    split_2_dev = {}
    split_3_tr = {}
    split_3_dev = {}
    count = 0
    docs_per_split = len(train_tokens_span) / 2
    for doc_name, values in train_tokens_span.items():
        count += 1
        if count <= docs_per_split:
            split_3_tr[doc_name] = values
            split_2_dev[doc_name] = values
        else:
            split_2_tr[doc_name] = values
            split_3_dev[doc_name] = values
    split_3_tr = {**split_3_tr, **dev_tokens_span}
    split_2_tr = {**split_2_tr, **dev_tokens_span}

    def write_data_for_splits(train_split, dev_split, write_path):
        create_directory(write_path)
        write_pickle(train_split, join_path(write_path, 'train-token_span.pkl'))
        write_pickle(dev_split, join_path(write_path, 'dev-token_span.pkl'))
        parse_from_list(txt_files=list(train_split.keys()), w_path=join_path(write_path, 'train.txt'),
                        doc_token_span_w_path=join_path(write_path, 'train-token_span.pkl'),
                        train_data_path=join_path(data_path, 'train/'),
                        dev_data_path=join_path(data_path, 'dev/'))
        parse_from_list(txt_files=list(dev_split.keys()), w_path=join_path(write_path, 'dev.txt'),
                        doc_token_span_w_path=join_path(write_path, 'dev-token_span.pkl'),
                        train_data_path=join_path(data_path, 'train/'),
                        dev_data_path=join_path(data_path, 'dev/'))

    # create token-span & bio files
    write_data_for_splits(split_2_tr, split_2_dev, write_path=join_path(base_path, 'data-strategy=2/'))
    write_data_for_splits(split_3_tr, split_3_dev, write_path=join_path(base_path, 'data-strategy=3/'))


def main():
    start_time = timeit.default_timer()
    paths = {'train': 'dataset/raw/train/',
             'dev': 'dataset/raw/dev/',
             'test': 'dataset/raw/test/'}
    for d_split, path in paths.items():
        print('**{}**'.format(d_split))
        if 'test' not in path:
            base_path = 'dataset/data-strategy=1/'
        else:
            base_path = 'dataset/'
        parse_fn = parse  # if 'test' not in path else parse_test
        parse_fn(path, join_path(base_path, d_split + '.txt'),
                 join_path(base_path, d_split + '-token_span.pkl'))
    create_data_splits(base_path='dataset/', span_path='dataset/data-strategy=1/', data_path='dataset/raw_temp/')
    end_time = timeit.default_timer()
    print('The code ran for {}m'.format(round((end_time - start_time) / 60.), 2))


if __name__ == '__main__':
    main()
