from utils.pharmaco_eval.evaluate import evaluation
from loader import prepare_sentence
from utils.model_utils import create_input, iobes_iob, zero_digits
from utils.utils import join_path, create_directory, delete_directory, read_pickle, file_exists
from data import get_ortho_feature
from model import Model

import numpy as np
import codecs
import argparse


def extract_entities(words, tags, start, end, majority_selection=False):
    entities = []
    phrase = ''
    last_tag = 'O'
    start_chunk = False
    end_chunk = False
    iter_idx = 0
    last_end_idx = None
    ph_st_idx = None
    ph_end_idx = None
    entity_labels = []
    for idx in range(len(words)):
        word, tag, st_idx, end_idx = words[idx], tags[idx], start[idx], end[idx]
        if last_tag == 'O' and tag.startswith('B'):
            start_chunk = True
            end_chunk = False
            ph_st_idx = st_idx

        if last_tag.startswith('B') and tag == 'O':
            end_chunk = True
            start_chunk = False
            ph_end_idx = last_end_idx

        if last_tag.startswith('I') and tag == 'O':
            end_chunk = True
            start_chunk = False
            ph_end_idx = last_end_idx

        if (last_tag.startswith('B') and tag.startswith('I')) or (
                last_tag.startswith('I') and tag.startswith('I')):
            end_chunk = False
            start_chunk = False

        if start_chunk:
            phrase = word
            start_chunk = False
            last_tag = tag
            entity_labels.append(tag.replace('B-', '').replace('I-', ''))
            if iter_idx == len(tags) - 1:
                end_chunk = False
                start_chunk = False
                # TODO: over-write everything under if with
                # end_chunk = True
                # ph_st_idx = st_idx
                # ph_end_idx = end_idx
                # entity_labels = []

        elif end_chunk:
            assert ph_st_idx is not None, 'ph_st_idx == None'
            assert ph_end_idx is not None, 'ph_end_idx == None'
            if majority_selection:
                _class = max(entity_labels, key=entity_labels.count)
            else:
                _class = entity_labels[0]
            entities.append({'text': phrase,
                             'class': last_tag.split('-')[-1],
                             'st_idx': ph_st_idx,
                             'end_idx': ph_end_idx})
            last_tag = tag
            end_chunk = False
            entity_labels = []
        elif tag == 'O' and last_tag == 'O':
            # outside of the phrase or chunk
            last_tag = tag
        else:
            # middle of the chunk
            phrase = phrase + ' ' + word
            entity_labels.append(tag.replace('B-', '').replace('I-', ''))
            last_tag = tag
            if iter_idx == len(tags) - 1:
                end_chunk = False
                start_chunk = False
        iter_idx += 1
        last_end_idx = end_idx

    return entities


def write_predictions(w_path, predictions):
    create_directory(w_path)
    count = 0
    for doc, res in predictions.items():
        f_path = join_path(w_path, '{}.ann'.format(doc))
        with codecs.open(f_path, 'w') as f:
            ent_count = 0
            for ent in res:
                ent_count += 1
                f.write('T{}\t{} {} {} {}\n'.format(ent_count, ent['class'], ent['st_idx'], ent['end_idx'],
                                                    ent['text']))
        count += 1
    print('{} files created: {}'.format(count, w_path))


# identify start of new tags and change I- to B- accordingly
def resolve_inconsistencies(y_preds):
    last_tag = 'O'
    for idx in range(len(y_preds)):
        if y_preds[idx].startswith('I-') and last_tag.startswith('O'):
            y_preds[idx] = y_preds[idx].replace('I-', 'B-')
        if y_preds[idx].startswith('I-') and last_tag.startswith('I-'):
            # but class is different
            if y_preds[idx] != last_tag:
                y_preds[idx] = y_preds[idx].replace('I-', 'B-')
        if y_preds[idx].startswith('I-') and last_tag.startswith('B-'):
            # but class is different
            if y_preds[idx].split('-')[-1] != last_tag.split('-')[-1]:
                y_preds[idx] = y_preds[idx].replace('I-', 'B-')
        last_tag = y_preds[idx]
    return y_preds


def extract_predictions_from_raw_text(model_path, tokens, pos):
    model = Model(model_path=model_path)
    parameters = model.parameters
    if 'language_model' not in parameters:
        parameters['language_model'] = False
    # Load reverse mappings
    word_to_id, char_to_id, tag_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]
    pos_to_id, ortho_to_id, segment_to_id = [
        {v: k for k, v in x.items()}
        for x in [model.id_to_pos, model.id_to_ortho, model.id_to_segment]
    ]
    word_to_id_1 = {v: k for k, v in model.id_to_word_1.items()}
    # Load the model
    _, f_eval = model.build(training=False, **parameters)
    model.reload()
    id_to_tag = model.id_to_tag
    sentence_cl = ' '.join(tokens)
    if parameters['lower']:
        sentence_cl = sentence_cl.lower()
    # Replace all digits with zeros
    if parameters['zeros']:
        sentence_cl = zero_digits(sentence_cl)
    tokens = sentence_cl.split(' ')
    ortho = [get_ortho_feature(w) for w in tokens]
    assert len(tokens) == len(pos) == len(ortho)
    input_dict = {'words': tokens, 'pos': pos, 'ortho': ortho}
    # Prepare input
    sentence = prepare_sentence(input_dict, word_to_id, char_to_id, pos_to_id, ortho_to_id,
                                segment_to_id, word_to_id_1,
                                lower=parameters['lower'])
    input = create_input(sentence, parameters, add_label=False)
    # Decoding
    if parameters['crf']:
        y_preds = np.array(f_eval(*input))[1:-1]
    else:
        y_preds = f_eval(*input).argmax(axis=1)
    y_preds = [id_to_tag[y_pred] for y_pred in y_preds]
    # Output tags in the IOB2 format
    if parameters['tag_scheme'] == 'iobes':
        y_preds = iobes_iob(y_preds)
    y_preds = resolve_inconsistencies(y_preds)
    return tokens, y_preds


def extract_tagger_predictions(model_path, span_path, output_path=None, f_eval=None,
                               parameters=None,
                               return_raw_predictions=False):
    assert file_exists(span_path)
    documents = read_pickle(span_path)
    if not f_eval:
        model = Model(model_path=model_path)
        parameters = model.parameters
        if 'language_model' not in parameters:
            parameters['language_model'] = False
        # Load reverse mappings
        word_to_id, char_to_id, tag_to_id = [
            {v: k for k, v in x.items()}
            for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
        ]
        pos_to_id, ortho_to_id, segment_to_id = [
            {v: k for k, v in x.items()}
            for x in [model.id_to_pos, model.id_to_ortho, model.id_to_segment]
        ]
        word_to_id_1 = {v: k for k, v in model.id_to_word_1.items()}
        # Load the model
        _, f_eval = model.build(training=False, **parameters)
        model.reload()
        id_to_tag = model.id_to_tag
    else:
        # load mappings
        mappings = read_pickle(join_path(model_path, 'mappings.pkl'))
        id_to_word = mappings['id_to_word']
        id_to_char = mappings['id_to_char']
        id_to_tag = mappings['id_to_tag']
        id_to_pos = mappings['id_to_pos']
        id_to_ortho = mappings['id_to_ortho']
        id_to_segment = mappings['id_to_segment']
        id_to_word_1 = mappings['id_to_word_1']
        # reverse mappings
        word_to_id, char_to_id, tag_to_id = [
            {v: k for k, v in x.items()}
            for x in [id_to_word, id_to_char, id_to_tag]
        ]
        pos_to_id, ortho_to_id, segment_to_id = [
            {v: k for k, v in x.items()}
            for x in [id_to_pos, id_to_ortho, id_to_segment]
        ]
        word_to_id_1 = {v: k for k, v in id_to_word_1.items()}
    predictions = {}
    docs_count = 0
    for doc_name, sentences in documents.items():
        for sentence in sentences:
            words = [span['word'] for span in sentence]
            start = [span['start'] for span in sentence]
            end = [span['end'] for span in sentence]
            pos = [span['pos'] for span in sentence]
            ortho = [get_ortho_feature(w) for w in words]
            doc_names = [doc_name] * len(words)
            input_dict = {'words': words, 'pos': pos, 'ortho': ortho, 'doc_names': doc_names}
            sentence_cl = ' '.join(words)
            if parameters['lower']:
                sentence_cl = sentence_cl.lower()
            # Replace all digits with zeros
            if parameters['zeros']:
                sentence_cl = zero_digits(sentence_cl)
            words = sentence_cl.split(' ')
            assert len(words) == len(start) == len(end)
            # Prepare input
            sentence = prepare_sentence(input_dict, word_to_id, char_to_id, pos_to_id, ortho_to_id,
                                        segment_to_id, word_to_id_1,
                                        lower=parameters['lower'])
            input = create_input(sentence, parameters, add_label=False)
            # Decoding
            if parameters['crf']:
                y_preds = np.array(f_eval(*input))[1:-1]
            else:
                y_preds = f_eval(*input).argmax(axis=1)
            y_preds = [id_to_tag[y_pred] for y_pred in y_preds]
            # Output tags in the IOB2 format
            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)
            if not return_raw_predictions:
                y_preds = resolve_inconsistencies(y_preds)
                entities = extract_entities(words, y_preds, start, end)
                if doc_name not in predictions:
                    predictions[doc_name] = []
                if len(entities) > 0:
                    predictions[doc_name] += entities
            else:
                if doc_name not in predictions:
                    predictions[doc_name] = {}
                    predictions[doc_name]['words'] = []
                    predictions[doc_name]['tags'] = []
                    predictions[doc_name]['start'] = []
                    predictions[doc_name]['end'] = []
                predictions[doc_name]['words'].append(words)
                predictions[doc_name]['tags'].append(y_preds)
                predictions[doc_name]['start'].append(start)
                predictions[doc_name]['end'].append(end)
        docs_count += 1
        if docs_count % 100 == 0:
            print('{} documents processed'.format(docs_count))

    if return_raw_predictions:
        return predictions
    else:
        write_predictions(output_path, predictions)


def evaluate_model(model_p, gt_p, token_span_p, f_eval=None, parameters=None, del_results=True, return_score=False):
    output_path = join_path(model_p, 'results/')
    extract_tagger_predictions(model_p, token_span_p, output_path=output_path, f_eval=f_eval, parameters=parameters)
    x = evaluation()
    eval_score = x.eval(output_path, gt_p, output_path)
    if del_results:
        delete_directory(output_path)
    if return_score:
        return eval_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Path of the trained model')
    parser.add_argument('-data', help='Path of the annotation files, needed to compute F1 score')
    parser.add_argument('-span', help='Path of the pickle file containing the token span for the whole test set')
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.span, del_results=False)


if __name__ == '__main__':
    main()
