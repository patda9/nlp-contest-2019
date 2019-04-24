import json
import re

from pythainlp.corpus.common import thai_words
from pythainlp.tokenize import word_tokenize

s_chars_pattern = re.compile(r"[\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{\|}~]")

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

def get_dataset(path='./dataset/'):
    c_file = open('%sans.txt' % (path))
    s_file = open('%sinput.txt' % (path), encoding='utf-8-sig')

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

def vectorize_token():
    pass

if __name__ == "__main__":
    classes, sentences = get_dataset()

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
