# To give credit where credit is do, this code is based off of the works found at the following sites:
#   https://keon.io/deep-q-learning/
#   https://github.com/keon/deep-q-learning/blob/master/ddqn.py
#   https://sergioskar.github.io/Deep_Q_Learning/
# I have taken the information from both the sites and combined it into my representation below.

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, LSTM, Dropout, AlphaDropout, PReLU
from keras.optimizers import Adam, RMSprop, Nadam
from keras import backend as K
import tensorflow as tf

from collections import deque
import random
from tqdm import tqdm

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        # self.epsilon_decay = 0.9995
        self.epsilon_decay = 0.995
        # self.learning_rate = 0.005
        self.learning_rate = 0.001
        # self.learning_rate = 10e-5
        # self.epsilon_decay = 0.99985
        # self.learning_rate = 10e-5
        self.batch_size = 32
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.count = 0
        self.is_finished = False
        self.loss = None


    def build_model(self):
        # Neural Net for Deep-Q learning Model
        # model = Sequential()
        # model.add(LSTM(24, return_sequences=False, input_shape=(1, self.state_size)))
        # model.add(Dropout(0.2))
        # model.add(Dense(self.action_size, activation='softmax'))
        # # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.learning_rate))
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))

        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='selu'))
        model.add(Dense(40, activation='selu'))
        model.add(Dense(40, activation='selu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        # model.compile(optimizer=Nadam(lr=self.learning_rate), loss='mse')

        # model = Sequential()
        # model.add(Dense(40, input_dim=self.state_size, activation='selu'))
        # model.add(AlphaDropout(0.1))
        # model.add(Dense(40, activation='selu'))
        # model.add(AlphaDropout(0.1))
        # model.add(Dense(40, activation='selu'))
        # model.add(AlphaDropout(0.1))
        # model.add(Dense(self.action_size, activation='softmax'))
        # # model.add(Dense(self.action_size, activation='softmax'))
        # # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action = None
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)

        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])

        return action

    def replay(self, batch_size):

        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = self.model.predict(state)
        #     if done:
        #         target[0][action] = reward
        #     else:
        #         # a = self.model.predict(next_state)[0]
        #         t = self.target_model.predict(next_state)[0]
        #         target[0][action] = reward + self.gamma * np.amax(t)
        #         # target[0][action] = reward + self.gamma * t[np.argmax(a)]
        #     self.model.fit(state, target, epochs=1, verbose=0)

        # # DQN
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         Q_next = self.model.predict(next_state)[0]
        #         target = (reward + self.gamma *np.amax(Q_next))
        #
        #     target_f = self.model.predict(state)
        #     target_f[0][action] = target
        #     #train network
        #     self.model.fit(state, target_f, epochs=1, verbose=0)

        # DDQN
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)
            testtarget = self.model.predict(state)
            target_next = self.model.predict(next_state)
            target_val = self.target_model.predict(next_state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * target_val[0][np.argmax(target_next)]

                # test = self.target_model.predict(next_state)[0]
                # testtarget[0][action] = reward + self.gamma * np.amax(test)
                #
                # if not np.array_equal(target, testtarget):
                #     print(target)
                #     print(testtarget)

            self.model.fit(state, target, epochs=1, verbose=0)
            self.loss = self.model.evaluate(state, target, verbose=0)

        self.decay_epsilon()

    def decay_epsilon(self):
        self.count += 1
        # if self.count % 20 == 0 and self.epsilon > self.epsilon_min:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name, use_weights=True):
        if use_weights:
            self.model.load_weights(name)
        else:
            self.model = load_model(name)

    def save(self, name):
        self.model.save_weights(name.replace('.h5', '-weights.h5'))
        self.model.save(name)
