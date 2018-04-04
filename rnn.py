import numpy as np
np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import seaborn as sns
from pylab import rcParams

tf.logging.set_verbosity(tf.logging.ERROR)

class RNN:
    # init RNN
    def __init__(self):
        self.model_path = 'models/realDonaldTrump.model'
        self.history_path = 'history/realDonaldTrump.hstr'
        self.sentence_length = 20
        self.step = 3
        self.epoch = 20
        self.source_path = 'data/realDonaldTrump.txt'

        text = open(self.source_path, encoding='utf-8').read().lower()

        # collect chars
        self.chars = sorted(list(set(text)))

        # [char => index]
        self.char_index = dict((c, i) for i, c in enumerate(self.chars))

        # [index => char]
        self.index_char = dict((i, c) for i, c in enumerate(self.chars))

        # load model
        self.load()
        if hasattr(self, 'model'):
            return

        # create sentences and next_char prediction array
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.sentence_length, self.step):
            # create array of sentences
            sentences.append(text[i: i + self.sentence_length])
            # create array of next chars
            next_chars.append(text[i + self.sentence_length])

        # create X and y vectors for training
        # X = [num_sentences, seq_length, num_chars]
        # Y = [num_sentences, num_chars]
        X = np.zeros((len(sentences), self.sentence_length, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for j, char in enumerate(sentence):
                # X[sentence index, char index in sentence, char index] = bool
                X[i, j, self.char_index[char]] = True
            # Y[sentence index, next char index] = bool
            y[i, self.char_index[next_chars[i]]] = True

        # build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.sentence_length, len(self.chars))))
        self.model.add(Dense(len(self.chars)))
        self.model.add(Activation('softmax'))

        # backpropagation training
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.history = self.model.fit(X, y, validation_split=0.05, batch_size=128, epochs=self.epoch,
                                      shuffle=True).history
        self.save()

    # load model and history
    def load(self):
        self.model = load_model(self.model_path)
        self.history = pickle.load(open(self.history_path, "rb"))

    # save model and history
    def save(self):
        self.model.save(self.model_path)
        pickle.dump(self.history, open(self.history_path, "wb"))

    # show plots
    def show_plots(self):
        # plots configuration
        plt.ion()
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 12, 5

        # accuracy figure
        accuracy = plt.figure(1)
        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

        # training and testing results
        plt.legend(['train', 'test'], loc='upper left')
        accuracy.show()

        # loss figure
        loss = plt.figure(2)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')

        # training and testing results
        plt.legend(['train', 'test'], loc='upper left')
        loss.show()

        # show plots
        plt.show(block=True)

    # prepare input
    def prepare_input(self, input):
        if len(input) < self.sentence_length:
            input = " " * (self.sentence_length - len(input)) + input
        if len(input) > self.sentence_length:
            input = input[-self.sentence_length:]
        formatted_input = np.zeros((1, self.sentence_length, len(self.chars)))
        for i, char in enumerate(input):
            formatted_input[0, i, self.char_index[char]] = 1.
        return formatted_input

    # predict next n most probable characters
    def get_next_char(self, prediction, n):
        prediction = np.asarray(prediction).astype('float64')
        return heapq.nlargest(n, range(len(prediction)), prediction.take)

    # predict one completion
    def get_completion(self, input):
        completion = ''
        while True:
            prediction = self.model.predict(self.prepare_input(input), verbose=0)[0]
            next_char = self.index_char[self.get_next_char(prediction, n=1)[0]]
            completion += next_char
            input += next_char
            if next_char == ' ' and len(completion) > 2 or len(completion) > 10:
                return completion

    # predict n completions
    def predict(self, input, n):
        prediction = self.model.predict(self.prepare_input(input), verbose=0)[0]
        next_char_indexes = self.get_next_char(prediction, n)
        return [self.index_char[idx] + self.get_completion(input + self.index_char[idx]) for idx in
                next_char_indexes]
