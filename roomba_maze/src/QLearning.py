import gym
import numpy as np
import random
from collections import defaultdict


class QLearning:

    def __init__(self, env, learning_rate, gamma, epsilon, num_of_episodes):
        # Environment Info
        self.env = env

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.opt_policy = defaultdict(lambda: 0)

        self.alpha = learning_rate
        self.initial_learning_rate = learning_rate
        self.min_learning_rate = 10e-4
        self.alpha_decay_rate = 0.95

        self.gamma = gamma

        self.epsilon = epsilon
        # self.greedy_rate = 1.0001
        self.greedy_rate = 1.2


        self.num_of_episodes = num_of_episodes

    # ================== #
    # ==== E Greedy ==== #
    # ================== #
    def e_greedy(self, state):
        """
        This runs the e-greedy algorithm that the book talks about

        :param state: The current state of the program
        :return: action: The chosen action to take
        """
        rand_num = random.random()
        if rand_num < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action

    # ===================== #
    # ==== Decay Alpha ==== #
    # ===================== #
    def decay_alpha(self, time_step):
        """
        This decays the alpha function. This function was referenced from the following sites:
        https://gist.github.com/malzantot/9d1d3fa4fdc4a101bc48a135d8f9a289
        https://www.tensorflow.org/api_docs/python/tf/train/inverse_time_decay
        https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay

        :return: The updated/decayed alpha value
        """
        # Exponential Decay
        decayed_alpha = max(self.min_learning_rate, self.initial_learning_rate * (self.alpha_decay_rate ** (time_step // 100)))

        # Inverse Time Decay
        # decayed_alpha = max(self.min_learning_rate, self.initial_learning_rate / (1 + self.decay_rate_value * time_step / 100))
        return decayed_alpha

    # ======================= #
    # ==== Decay Epsilon ==== #
    # ======================= #
    def greedfiy(self, time_step):
        """
        This slowly greedyifies the algorithm if used

        :return: greedy_e: The updated epsilon value.
        """
        # greedy_e = min(0.98, 0.1 ** (1.0 / (1.0 + time_step)))
        greedy_e = max(0.1, min(0.95, self.epsilon ** self.greedy_rate))
        return greedy_e


    # ================= #
    # ==== Q Learn ==== #
    # ================= #
    def q_learn(self):
        """
        Runs the Q Learning tabular model

        :return: q_table: The Updated Q table
        """
        episode_i = 0
        avg_reward_list = []

        while episode_i < self.num_of_episodes:
            self.env.render()

            # Initialize S
            state = tuple(self.env.reset())
            # tuple_state = self.map_env_to_discrete_env(state)

            self.alpha = self.decay_alpha(episode_i)

            episode_step_i = 0  # The count for the inner loop
            avg_reward = 0

            # loop through the steps within an episode
            while True:
                self.env.render()
                # Choose action A from state S using policy derived from Q (e.g., e-greedy)
                action = self.e_greedy(state)

                # Take action A, observe R, next_state S
                next_state, reward, done, info = self.env.step(action)
                next_state = tuple(next_state)

                # Q(S,A) = Q(S,A) + alpha(R + gamma * max a Q(S', a) − Q(S,A))
                self.q_table[state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action])

                # S  = S'
                state = next_state

                avg_reward += reward

                if done:
                    avg_reward_list.append(avg_reward)

                    print('Episode: ', episode_i, ' Episode Reward = ', avg_reward, ' LR: ', self.alpha, ' GAMMA: ', self.gamma, ' EP: ', self.epsilon)
                    break

                episode_step_i += 1

            if episode_i % 10 == 0:
                self.epsilon = self.greedfiy(episode_i)

            if episode_i % 100 == 0:
                with open('./qtable_{}epochs.txt'.format(episode_i), 'w') as output:
                    output.write(str(self.q_table))

            episode_i += 1

        with open('./qtable_{}epochs.txt'.format(episode_i), 'w') as output:
            output.write(str(self.q_table))

        return self.q_table, avg_reward_list