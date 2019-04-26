import json
import re

import numpy as np

from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import word_tokenize

s_chars_pattern = re.compile(r"[\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{\|}~“”!]")

def break_english(s, pattern=s_chars_pattern):
    for i in range(len(s)):
        try:
            s[i].encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            english_chars = ''
            for c in s[i]:
                try:
                    if(c.isdigit() or c in pattern.findall(c)):
                        pass
                    else:
                        c.encode(encoding='utf-8').decode('ascii')
                        english_chars += c
                except UnicodeDecodeError:
                    pass

            s[i] = s[i].replace(english_chars, '')

            if(english_chars):
                s.insert(i + 1, english_chars)
                break
    return s

def get_dataset(l='', path='./dataset/', s=''):
    c_file = open('%s%s' % (path, l))
    s_file = open('%s%s' % (path, s), encoding='utf-8-sig')

    classes = []
    for line in c_file:
        head = re.compile(r"([0-9]+::)|\n|[^A-Z]")
        classes.append(re.sub(head, '', line))

    sentences = []
    for line in s_file:
        head = re.compile(r"([0-9]+::)|\n")
        sentences.append(re.sub(head, '', line))

    return classes, sentences

def get_sentence_labels(path='./dataset/tokenized_dataset/labels.txt'):
    with open(path) as f:
        labels = json.load(f)
    
    return labels

def get_tkned_sentences(path='./dataset/tokenized_dataset/sentences.json'):
    with open(path, encoding='utf-8-sig') as f:
        sentences = json.load(f)
        
    return sentences

def preprocess_sentence(s, pattern=s_chars_pattern):
    if(type(s) == type('')):
        matched_chars = pattern.findall(s)
        prep_s = ''
        for c in s:
            if(c == '–'):
                c = ' '
            if(not(c in matched_chars)):
                prep_s += c
    elif(type(s) == type([])):
        prep_s = []
        for i in range(len(s)):
            prep_t = ''
            for t in s[i]:
                matched_chars = pattern.findall(t)
                for c in t:
                    if(not(c in matched_chars)):
                        prep_t += c
            if(prep_t):
                prep_s.append(prep_t)

    return prep_s

def tokenize_sentence(s):
    tkned_s = word_tokenize(s, engine='newmm')
    
    return tkned_s

def vectorize_tokens(sentences, return_vocab_wvs=0, wv_path='D:/Users/Patdanai/th-qasys-db/fasttext_model/cc.th.300.vec', verbose=0):
    wv_fp = open(wv_path, encoding='utf-8-sig')

    MAX_SEQ_LENGTH = len(max(sentences, key=len))
    wvl = 300
    
    vocabs = set([t for s in sentences for t in s])

    count = 0
    vocab_wvs = {}
    for line in wv_fp:
        if(count > 0):
            line = line.split()
            if(line[0] in vocabs):
                if(verbose):
                    print('found %s %s' % (line[0], count))
                vocab_wvs[line[0]] = line[1:]
        count += 1

    wv = np.zeros((len(sentences), MAX_SEQ_LENGTH, wvl))

    count = 0
    for s in sentences:
        t_count = 0
        for t in s:
            try:
                wv[count, MAX_SEQ_LENGTH - 1 - t_count, :] = vocab_wvs[t]
                t_count += 1
            except:
                pass
        count += 1
    
    if(return_vocab_wvs):
        return wv, vocab_wvs

    return wv, None

if __name__ == "__main__":
    classes, sentences = get_dataset(l='ans.txt', s='input.txt')

    # write tokenized sentences file
    white_space_match = re.compile(r"[ \s]")
    for i in range(len(sentences)):
        sentences[i] = preprocess_sentence(sentences[i])
        sentences[i] = tokenize_sentence(sentences[i])
        sentences[i] = break_english(sentences[i])
        sentences[i] = preprocess_sentence(sentences[i], pattern=white_space_match)

    f = open('./dataset/tokenized_dataset/sentences.json', 'w', encoding='utf-8-sig')
    json.dump(sentences, f, ensure_ascii=0)

    # write labels file
    labels = []
    for l in classes:
        if(l == 'H'):
            c = 0
        elif(l == 'P'):
            c = 1
        elif(l == 'M'):
            c = 2
        labels.append(c)

    f = open('./dataset/tokenized_dataset/labels.txt', 'w')
    json.dump(labels, f)
