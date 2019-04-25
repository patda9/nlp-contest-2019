import numpy as np

from keras.backend import mean
from keras.layers import BatchNormalization, Bidirectional, Dense, Flatten, GRU, Input, Lambda, Masking, multiply, Permute, RepeatVector
from keras.models import Model
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import utitlities as utils

from sklearn.metrics import confusion_matrix

def attention_layer(inputs, time_step, single_att_vec=0):
    a = Lambda(lambda x: x, output_shape=lambda s: s)(inputs)
    a = Permute((2, 1))(a)
    a = Dense(time_step, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

if __name__ == "__main__":
    labels = np.array(utils.get_sentence_labels())
    sentences = utils.get_tkned_sentences()

    MAX_SEQ_LENGTH = len(max(sentences, key=len))
    vocabs = set([t for s in sentences for t in s])

    wv_fp = open('D:/Users/Patdanai/th-qasys-db/fasttext_model/cc.th.300.vec', encoding='utf-8-sig')

    count = 0
    vocab_wvs = {}
    for line in wv_fp:
        if(count > 0):
            line = line.split()
            if(line[0] in vocabs):
                print('found %s %s' % (line[0], count))
                vocab_wvs[line[0]] = line[1:]
        count += 1

    wvl = 300
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
    
    loc = np.random.permutation(wv.shape[0])
    wv = wv[loc]
    labels = labels[loc]
    sentences = np.array(sentences)
    sentences = sentences[loc]

    input_layer = Input(shape=(MAX_SEQ_LENGTH, wvl))
    mask = Masking(mask_value=0., input_shape=(MAX_SEQ_LENGTH, wvl))(input_layer)
    gru_1 = Bidirectional(GRU(64, dropout=.2, recurrent_dropout=.1, return_sequences=1))(mask)
    batch_norm_1 = BatchNormalization()(gru_1)
    attention_1 = attention_layer(batch_norm_1, MAX_SEQ_LENGTH)
    attention_1 = Lambda(lambda x: x, output_shape=lambda s: s)(attention_1)
    attention_1 = Flatten()(attention_1)
    output_layer = Dense(3, activation='softmax')(attention_1)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()

    loaded = 0
    if(not(loaded)):
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])
    
    history = model.fit(wv, to_categorical(labels, num_classes=3), batch_size=wv.shape[0], epochs=256, validation_split=.2)
    
    from datetime import datetime
    tsp = datetime.now()
    tsp = str(tsp)
    tsp = tsp.replace(':', '-')
    model.save('./model/%s.h5' % tsp)

    x_test = wv[int(.8*len(wv)):, :]
    y_test = labels[int(.8*len(wv)):]
    verifying_s = sentences[int(.8*len(wv)):]
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions.argmax(axis=1))
    print(cm)

    with open('./result/verification.txt', 'w', encoding='utf-8-sig') as f:
        for i in range(predictions.shape[0]):
            f.writelines('s:%s, l:%s, pred:%s/n' % (verifying_s[i], y_test[i], predictions[i]))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
