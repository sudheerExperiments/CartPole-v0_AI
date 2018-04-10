#!/usr/bin/python3

# Issues:
# 1. NotImplementedError: abstract
# https://github.com/openai/gym/issues/775
# Solution: Uninstall current version and use pyglet version: 1.2.4

# References:
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/

import gym
import numpy as np
import random
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


class RandomGame(object):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.done = False
        self.moves_count = 0
        # A vector with parameters, Position of the cart, Velocity of the cart, Angle of the pole,  Velocity at its tip
        self.observation = self.env.reset()

    def make_random_move(self):
        while not self.done:
            # Comment if displaying move is not needed
            self.env.render()
            self.moves_count += 1

            action = self.env.action_space.sample()
            self.observation, reward, done, _ = self.env.step(action)

            if done:
                break

        print("No. of moves: {}".format(self.moves_count))


class SmartGame(object):
    def __init__(self):
        self.done = False
        self.moves_count = 0
        # A vector with parameters, Position of the cart, Velocity of the cart, Angle of the pole,  Velocity at its tip
        # https://github.com/yandexdataschool/Practical_RL/issues/9
        self.env = gym.make("CartPole-v0").env
        self.env.reset()

        self.ntrain_samples = 10000
        self.goal = 500
        self.learning_rate = 0.001

        # Observations
        self.X_train = []
        # Actions
        self.Y_train = []

    def generate_training_data(self, use_model=False):
        # Reset the game to initial position
        self.env.reset()
        # All scores
        scores = []
        # Scores above certain threshold
        accepted_scores = []
        score_requirement = 50

        training_data = []

        if use_model:
            training_data = np.load("data/train_data.npy")

        if not use_model:
            for temp in range(self.ntrain_samples):
                score = 0

                # Stores of all observation and moves
                game_memory = []
                # previous observation
                prev_observation = []

                for temp1 in range(self.goal):
                    # First move
                    action = random.randrange(0, 2)
                    observation, reward, done, info = self.env.step(action)

                    if len(prev_observation) > 0:
                        game_memory.append([prev_observation, action])

                    prev_observation = observation
                    score += reward
                    if done:
                        # Reset when game is over, otherwise step() won't work, the whole model fails
                        self.env.reset()
                        break

                if score >= score_requirement:
                    output = []
                    accepted_scores.append(score)
                    # One hot encoding
                    for sample in game_memory:
                        if sample[1] == 1:
                            output = [0, 1]
                        elif sample[1] == 0:
                            output = [1, 0]

                        training_data.append([sample[0], output])

                self.env.reset()

                scores.append(score)

            # Save training data
            np.save('data/train_data', training_data)

            print("Average accepted scores: ", mean(accepted_scores))
            print('Median accepted scores:', median(accepted_scores))
            print(Counter(accepted_scores))

        return training_data

    # http://tflearn.org/
    def model(self, input_size, use_model=False):
        x_train = input_data(shape=[None, input_size, 1], name='input')

        hidden_layer1 = fully_connected(x_train, 128, activation='relu')
        hidden_layer1 = dropout(hidden_layer1, 0.8)

        hidden_layer2 = fully_connected(hidden_layer1, 256, activation='relu')
        hidden_layer2 = dropout(hidden_layer2, 0.8)

        hidden_layer3 = fully_connected(hidden_layer2, 512, activation='relu')
        hidden_layer3 = dropout(hidden_layer3, 0.8)

        hidden_layer4 = fully_connected(hidden_layer3, 256, activation='relu')
        hidden_layer4 = dropout(hidden_layer4, 0.8)

        hidden_layer5 = fully_connected(hidden_layer4, 128, activation='relu')
        hidden_layer5 = dropout(hidden_layer5, 0.8)

        network = fully_connected(hidden_layer5, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self.learning_rate, loss='categorical_crossentropy',
                             name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')

        # https://github.com/tflearn/tflearn/issues/39
        if use_model:
            model.load('data/sample.model')

        return model

    def train(self, train_data, model=False):
        self.X_train = np.array([temp[0] for temp in train_data]).reshape(-1, len(train_data[0][0]), 1)
        self.Y_train = [temp[1] for temp in train_data]

        # Use pre-trained model
        # https://github.com/tflearn/tflearn/issues/39
        if model:
            model = self.model(len(self.X_train[0]), True)

        if not model:
            model = self.model(len(self.X_train[0]))
            model.fit({'input': self.X_train}, {'targets': self.Y_train}, n_epoch=3, snapshot_step=500, show_metric=True)

        return model

    def test(self, model):
        observation = self.env.reset()
        scores = []
        choices = []

        for each_game in range(10):
            score = 0
            game_memory = []
            prev_obs = []

            for temp in range(self.goal):
                # See the visualization
                # self.env.render()

                # First move
                if len(prev_obs) == 0:
                    action = random.randrange(0, 2)
                else:
                    action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

                choices.append(action)
                new_observation, reward, done, info = self.env.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score += reward

                if done:
                    # Reset when game is over, otherwise step() won't work, the whole model fails
                    self.env.reset()
                    break

            scores.append(score)

        print('Average Score:', sum(scores) / len(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))


if __name__ == "__main__":
    # Play random game
    # obj = RandomGame()
    # obj.make_random_move()

    obj = SmartGame()
    # To use pre - trained model pass True
    training_data = obj.generate_training_data(True)
    # To use pre - trained model pass True
    returned_model = obj.train(training_data, True)

    # save model
    # returned_model.save("data/sample.model")

    obj.test(returned_model)