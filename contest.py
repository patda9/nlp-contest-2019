import classifier
import utitlities

import json
import matplotlib.pyplot as plt
import sys
import re

import numpy as np

from keras.models import load_model

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def run(argv):
    # l = argv[0]
    # s = argv[1]

    additional_labels = np.array(utitlities.get_sentence_labels('./dataset/tokenized_additional_sentences/additional_labels.txt'))
    additional_labels = additional_labels.reshape(len(additional_labels), 1)
    additional_sentences = utitlities.get_tkned_sentences('./dataset/tokenized_additional_sentences/addtional_sentences.json')

    # raw_labels, raw_sentences = utitlities.get_dataset(path='./dataset/', l=l, s=s)

    # white_space_match = re.compile(r"[ \s]")
    # for i in range(len(raw_sentences)):
    #     raw_sentences[i] = utitlities.preprocess_sentence(raw_sentences[i])
    #     raw_sentences[i] = utitlities.tokenize_sentence(raw_sentences[i])
    #     raw_sentences[i] = utitlities.break_english(raw_sentences[i])
    #     raw_sentences[i] = utitlities.preprocess_sentence(raw_sentences[i], pattern=white_space_match)
    
    # sentences = raw_sentences

    # labels = np.array(raw_labels)
    # labels[labels == 'H'] = 0
    # labels[labels == 'P'] = 1
    # labels[labels == 'M'] = 2
    # labels = labels.astype(int)
    
    # print(sentences)
    # print(labels)

    # wv, _ = utitlities.vectorize_tokens(sentences, wv_path='C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec', return_vocab_wvs=1)
    # print(wv.shape)
    # print(_)

    wv, _ = utitlities.vectorize_tokens(additional_sentences, max_seq_length=78, wv_path='C:/Users/Patdanai/Desktop/261499-nlp/lab/cc.th.300.vec', return_vocab_wvs=1)

    model = load_model('./model/2019-04-25 16-13-48.283617.h5')
    model.summary()

    x_test = wv
    print(wv.shape)
    y_test = additional_labels
    print(y_test.shape)
    # verifying_s = additional_sentences[int(.8*len(wv)):]
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions.argmax(axis=1))
    print(cm)

    class_names = np.array(['Horn', 'Person', 'Mountain'], dtype='U10')

    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, predictions.argmax(axis=1), classes=class_names,
                        title='Confusion matrix, without normalization')
    plt.show()

if __name__ == "__main__":
    run(sys.argv[1:3])