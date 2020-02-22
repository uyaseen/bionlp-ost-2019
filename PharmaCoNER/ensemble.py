from utils.pharmaco_eval.evaluate import evaluation
from utils.model_utils import iobes_iob, get_current_time
from evaluate import extract_tagger_predictions, extract_entities, write_predictions, resolve_inconsistencies

import timeit
import argparse


# mode at token level
def get_mode(d1, d2, d3):
    result = []
    assert len(d1) == len(d2) == len(d3)
    for idx in range(len(d1)):
        d_list = [d1[idx], d2[idx], d3[idx]]
        vote = max(d_list, key=d_list.count)
        result.append(vote)
    return result


def aggregate_ensemble_predictions(ensemble_models, span, predictions_path):
    ensemble_predictions = {}
    for model_name, model_path in ensemble_models.items():
        print('Model: {}'.format(model_name))
        ensemble_predictions[model_name] = extract_tagger_predictions(model_path=model_path,
                                                                      span_path=span,
                                                                      return_raw_predictions=True)
    results = {}
    for doc_name in ensemble_predictions['m1'].keys():
        m1, m2, m3 = ensemble_predictions['m1'][doc_name], \
                     ensemble_predictions['m2'][doc_name], \
                     ensemble_predictions['m3'][doc_name]

        for sentence_idx in range(len(m1['words'])):
            assert len(m1['words'][sentence_idx]) == len(m2['words'][sentence_idx]) == len(m3['words'][sentence_idx])
            tags1, tags2, tags3 = m1['tags'][sentence_idx], m2['tags'][sentence_idx], m3['tags'][sentence_idx]
            tags1 = iobes_iob(tags1)
            tags2 = iobes_iob(tags2)
            tags3 = iobes_iob(tags3)
            tags_vote = get_mode(tags1, tags2, tags3)
            tags_vote = resolve_inconsistencies(tags_vote)
            entities = extract_entities(m1['words'][sentence_idx],
                                        tags_vote,
                                        m1['start'][sentence_idx],
                                        m1['end'][sentence_idx],
                                        doc_name)
            if doc_name not in results:
                results[doc_name] = []
            if len(entities) > 0:
                results[doc_name] += entities

    write_predictions(predictions_path, results)


def main():
    start_time = timeit.default_timer()
    parser = argparse.ArgumentParser()
    parser.add_argument('-model1', help='Path of the 1st trained model')
    parser.add_argument('-model2', help='Path of the 2nd trained model')
    parser.add_argument('-model3', help='Path of the 3rd trained model')
    parser.add_argument('-data', help='Path of the annotation files, needed to compute F1 score')
    parser.add_argument('-span', help='Path of the pickle file containing the token span for the whole test set')
    args = parser.parse_args()
    ensemble_models = {'m1': args.model1, 'm2': args.model2, 'm3': args. model3}
    output_path = 'models/ensemble_{}'.format(get_current_time())
    aggregate_ensemble_predictions(ensemble_models, args.span, output_path)
    x = evaluation()
    x.eval(output_path, args.data, output_path)
    end_time = timeit.default_timer()
    print('Ensemble evaluation took {}m'.format(round((end_time - start_time) / 60.), 2))


if __name__ == '__main__':
    main()
