import classifier
import utitlities

import json
import sys
import re

import numpy as np

from keras.models import load_model

def run(argv):
    l = argv[0]
    s = argv[1]
    raw_labels, raw_sentences = utitlities.get_dataset(path='./dataset/', l=l, s=s)

    white_space_match = re.compile(r"[ \s]")
    for i in range(len(raw_sentences)):
        raw_sentences[i] = utitlities.preprocess_sentence(raw_sentences[i])
        raw_sentences[i] = utitlities.tokenize_sentence(raw_sentences[i])
        raw_sentences[i] = utitlities.break_english(raw_sentences[i])
        raw_sentences[i] = utitlities.preprocess_sentence(raw_sentences[i], pattern=white_space_match)
    
    sentences = raw_sentences

    labels = np.array(raw_labels)
    labels[labels == 'H'] = 0
    labels[labels == 'P'] = 1
    labels[labels == 'M'] = 2
    labels = labels.astype(int)
    
    print(sentences)
    print(labels)

    wv, _ = utitlities.vectorize_tokens(sentences, wv_path='C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec', return_vocab_wvs=1)
    print(wv.shape)
    print(_)

    model = load_model('./model/2019-04-25 16-13-48.283617.h5')
    model.summary()

if __name__ == "__main__":
    run(sys.argv[1:3])