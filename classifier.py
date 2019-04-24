import numpy as np

from keras.layers import BatchNormalization, Dense, Input, Masking
from keras.models import Model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import utitlities as utils

from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    labels = np.array(utils.get_sentence_labels())
    sentences = np.array(utils.get_tkned_sentences())

    vocabs = set([t for s in sentences for t in s])
    bows = np.zeros((len(sentences), len(vocabs)))

    for i in range(0, len(sentences)):
        count = 0
        for j in range(0, len(sentences[i])):
            k = 0
            for w in vocabs:
                if(sentences[i][j] == w):
                    bows[i][k] = bows[i][k] + 1
                    count = count + 1
                k += 1
        bows[i] = bows[i] / count # normalize with its occurrences
    
    loc = np.random.permutation(bows.shape[0])
    bows = bows[loc]
    labels = labels[loc]
    sentences = sentences[loc]

    input_layer = Input(shape=((len(vocabs), )))
    dense_1 = Dense(64, activation='tanh')(input_layer)
    batch_norm_1 = BatchNormalization()(dense_1)
    dense_2 = Dense(64, activation='tanh')(batch_norm_1)
    batch_norm_2 = BatchNormalization()(dense_2)
    output_layer = Dense(3, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    loaded = 0
    if(not(loaded)):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mean_absolute_error'])
    
    history = model.fit(bows, to_categorical(labels, num_classes=3), batch_size=1024, epochs=64, validation_split=.2)

    from datetime import datetime
    tsp = datetime.now()
    tsp = str(tsp)
    tsp = tsp.replace(':', '-')
    model.save('./model/%s' % tsp)

    x_test = bows[int(.8*len(bows)):, :]
    y_test = labels[int(.8*len(bows)):]
    verifying_s = sentences[int(.8*len(bows)):]
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions.argmax(axis=1))
    print(cm)

    with open('./result/verification.txt', 'w', encoding='utf-8-sig') as f:
        for i in range(predictions.shape[0]):
            f.writelines('s:%s, l:%s, pred:%s\n' % (verifying_s[i], labels[i], predictions[i]))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
