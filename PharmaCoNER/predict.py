from data import tokenize
from Med_Tagger import Med_Tagger
from evaluate import extract_predictions_from_raw_text

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Path of the trained model')
    parser.add_argument('-text', help='Extract entities from this text')
    args = parser.parse_args()
    med_tagger = Med_Tagger()  # Starts a docker image in background
    tokens, pos = tokenize(args.text, med_tagger)
    tokens, pos = tokens[0], pos[0]
    tokens, tags = extract_predictions_from_raw_text(args.model, tokens, pos)
    for tkn, tg in zip(tokens, tags):
        print('{}\t{}'.format(tkn, tg))
    del med_tagger


if __name__ == '__main__':
    main()
