import os

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once per notebook.
!pip install -U -q PyDrive
!pip install -U -q scikit-image
!pip install -U -q numpy
!pip install -U -q pandas
!pip install -U -q folium
!pip install -U -q imgaug
!pip install -U -q bresenham
!pip install -U -q tqdm

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

fid = drive.ListFile({'q': "title='mazelab_0_2_0.zip'"}).GetList()[0]['id']
f = drive.CreateFile({'id': fid})
f.GetContentFile('mazelab_0_2_0.zip')

path_to_zip_file = 'mazelab_0_2_0.zip'
directory_to_extract_to = './'

import zipfile

zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
zip_ref.extractall(directory_to_extract_to)
zip_ref.close()

print(os.getcwd())
os.chdir('mazelab_0_2_0')
print(os.getcwd())
!pip install -e .
os.chdir('../')
print(os.getcwd())

import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from mazelab2 import BaseMaze
from mazelab2 import BaseEnv
from mazelab2 import Object
from mazelab2 import DeepMindColor as color
from mazelab2.generators import random_shape_maze
from mazelab2 import VonNeumannMotion
from mazelab2.solvers import dijkstra_solver
import gym
from gym.spaces import Box
from gym.spaces import Discrete
from gym.wrappers import Monitor
from math import sqrt, cos, sin, atan2, pi
from bresenham import bresenham
import random
import time
from tqdm import tqdm
from collections import defaultdict
from collections import deque
import ast
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, LSTM, Dropout
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf



# =============================== #
# ==== Roomba Maze Generator ==== #
# =============================== #
class RoombaMazeGenerator2(BaseMaze):

    # ============== #
    # ==== Init ==== #
    # ============== #
    def __init__(self, width=50, height=50, max_num_shapes=50, max_size=8, seed=None):
        # super(RoombaMazeGenerator, self).__init__()
        self.maze_layout = random_shape_maze(width, height, max_num_shapes, max_size, allow_overlap=False, shape=None, seed=seed)
        super().__init__()
        self.maze_width = width
        self.maze_height = height
        self.num_maze_shapes = max_num_shapes
        self.max_size = max_size

    @property
    def size(self):
        return self.maze_layout.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.maze_layout == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.maze_layout == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        # agent = Object('agent', 2, color.agent, True, [])
        goal = Object('goal', 3, color.goal, False, [])
        # lidar = Object('lidar', 4, (255, 255, 204), False, [])
        return free, obstacle, agent, goal



class RoombaEnv2(BaseEnv):

    # ============== #
    # ==== Init ==== #
    # ============== #
    def __init__(self, training_type='escape', num_of_agents=1, width=50, height=50, max_num_shapes=50, max_size=8, seed=None):
        super().__init__()

        self.env_id = 'RandomShapeMaze-v0'
        self.training_type = training_type
        print('Training to ', self.training_type, ' an area!')

        self.motions = VonNeumannMotion()
        self.start_idx = [[1, 1]]
        self.goal_idx = [[height - 2, width - 2]]
        self.maze = None
        self.num_agents = num_of_agents

        self.build_maze(width, height, max_num_shapes, max_size, seed=seed)

        # self.lidar = Lidar(self.start_idx[0], self)
        self.lidars = [Lidar(self.start_idx[0], self)] * num_of_agents
        # self.slam = RMHC_SLAM(LaserModel(), self.maze.maze_width, 5)
        # self.mapbytes = bytearray(self.maze.maze_width * self.maze.maze_height)
        self.seen_lidar_scans = [[]] * num_of_agents
        self.num_of_invalid_moves = 0
        self.num_of_seen_moves = 0
        self.num_of_new_moves = 0

        self.positions_left_to_visit = [self.maze.objects.free.positions.tolist()] * self.num_agents

    # ============== #
    # ==== Step ==== #
    # ============== #
    def step(self, action, agent=0):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        reward = 0

        motion = self.motions[action]
        # current_position = self.maze.objects.agent.positions[0]
        current_position = self.maze.objects.agent.positions[agent]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self.is_valid(new_position)
        if valid:
            # self.maze.objects.agent.positions = [new_position]
            self.maze.objects.agent.positions[agent] = new_position
            # self.lidar.move([new_position[1], new_position[0]])
            self.lidars[agent].move([new_position[1], new_position[0]])
        else:
            new_position = current_position

        # self.lidar.scan(self.maze.maze_layout, self.maze.objects.agent)
        self.lidars[agent].scan(self.maze.maze_layout, self.maze.objects.agent.positions[agent])

        found_new_spot = False
        # if self.lidar.angle_dist_coord_pt not in self.seen_lidar_scans:
        # if self.lidar.small_angle_dist_list not in self.seen_lidar_scans:
        if self.lidars[agent].small_angle_dist_list not in self.seen_lidar_scans[agent]:
            found_new_spot = True
            # self.seen_lidar_scans.append(self.lidar.angle_dist_coord_pt)
            # self.seen_lidar_scans.append(self.lidar.small_angle_dist_list)
            self.seen_lidar_scans[agent].append(self.lidars[agent].small_angle_dist_list)

        # Get Reward and Check if agent is done
        reward, done = self.get_reward(new_position, agent, found_new_spot, valid)

        # observation = deepcopy(self.lidar.small_angle_dist_list)
        observation = deepcopy(self.lidars[agent].small_angle_dist_list)
        if found_new_spot:
            observation.append(1)  # Has seen a new spot
        else:
            observation.append(-1)  # Has seen spot before

        # return self.lidar.angle_dist_coord_pt, reward, done, {}
        # return self.lidar.small_angle_dist_list, reward, done, {}
        return observation, reward, done, {}

    # ==================== #
    # ==== Get Reward ==== #
    # ==================== #
    def get_reward(self, new_position, agent, found_new_spot, valid):
        reward = 0
        done = False

        # Reward structure from getting from Point A to Point B
        if self.training_type == 'escape':
            if self.is_goal(new_position):
                reward = +10  # Reward used for DQN to find path in maze
                done = True
            elif not valid:
                reward = -10  # Reward used for DQN to find path in maze
                self.num_of_invalid_moves += 1
            elif not found_new_spot:
                reward = -0.5  # Reward used for DQN to find path in maze
                self.num_of_seen_moves += 1

            elif found_new_spot:
                reward = +0.3  # Reward used for DQN to find path in maze
                self.num_of_new_moves += 1
            else:
                reward = -0.01

        # Reward structure for searching all of gridword
        elif self.training_type == 'sweep':
            if self.searched_complete_grid(new_position, agent):
                if found_new_spot:
                    reward = +0.3
                    self.num_of_new_moves += 1
                reward += +10
                done = True
            elif not valid:
                reward = -10
                self.num_of_invalid_moves += 1
            elif not found_new_spot:
                reward = -0.5
                self.num_of_seen_moves += 1
            elif found_new_spot:
                reward = +0.3
                self.num_of_new_moves += 1

        return reward, done

    # =============== #
    # ==== Reset ==== #
    # =============== #
    def reset(self, agent=0):
        self.num_of_invalid_moves = 0
        self.num_of_seen_moves = 0
        self.num_of_new_moves = 0
        self.positions_left_to_visit = [self.maze.objects.free.positions.tolist()] * self.num_agents

        # while True:
        #     rand_x = random.randint(self.maze.size[1]/2, self.maze.size[1] -2)
        #     rand_y = random.randint(self.maze.size[0] / 2, self.maze.size[0] - 2)
        #     if self.maze.maze_layout[rand_y][rand_x] != 1:
        #         self.goal_idx = [[rand_y, rand_x]]
        #         break

        # self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.agent.positions = [self.start_idx[0]] * self.num_agents
        self.maze.objects.goal.positions = self.goal_idx

        # self.lidar.scan(self.maze.maze_layout, self.maze.objects.agent)
        self.lidars[agent].scan(self.maze.maze_layout, self.maze.objects.agent.positions[agent])
        # self.seen_lidar_scans = []
        self.seen_lidar_scans[agent] = []
        # self.seen_lidar_scans.append(deepcopy(self.lidar.angle_dist_coord_pt))
        # self.seen_lidar_scans.append(deepcopy(self.lidar.small_angle_dist_list))
        self.seen_lidar_scans[agent].append(deepcopy(self.lidars[agent].small_angle_dist_list))

        # degree_0_dist = self.lidar.angle_dist_coord_pt[0]
        # degree_90_dist = self.lidar.angle_dist_coord_pt[90]
        # degree_180_dist = self.lidar.angle_dist_coord_pt[180]
        # degree_270_dist = self.lidar.angle_dist_coord_pt[270]
        #
        # deg_0_pos = [[self.start_idx[0][0], self.start_idx[0][1] + index] for index in range(self.start_idx[0][1], int(degree_0_dist))]
        # deg_90_pos = [[self.start_idx[0][0] - index, self.start_idx[0][1]] for index in range(self.start_idx[0][0], int(degree_90_dist))]
        # deg_180_pos = [[self.start_idx[0][0], self.start_idx[0][1] - index] for index in range(self.start_idx[0][1], int(degree_180_dist))]
        # deg_270_pos = [[self.start_idx[0][0] + index, self.start_idx[0][1]] for index in range(self.start_idx[0][0], int(degree_270_dist))]
        #
        # self.maze.objects.goal.positions = self.goal_idx
        # print(deg_0_pos + deg_90_pos + deg_180_pos + deg_270_pos)
        # self.maze.objects.lidar.positions = deg_0_pos + deg_90_pos + deg_180_pos + deg_270_pos

        # return self.lidar.angle_dist_coord_pt
        # return self.lidar.small_angle_dist_list
        # observation = deepcopy(self.lidar.small_angle_dist_list)
        observation = deepcopy(self.lidars[agent].small_angle_dist_list)
        observation.append(1)
        return observation

    # ================== #
    # ==== Is Valid ==== #
    # ================== #
    def is_valid(self, position):
        # Check and make sure agent isn't off grid in x or y postion
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]

        # Make sure the position an agent is going to is free to move to.
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        # Make sure position passes ALL 3 validation criteria
        is_valid = nonnegative and within_edge and passable

        return is_valid

    # ================================ #
    # ==== Searched Complete Grid ==== #
    # ================================ #
    def searched_complete_grid(self, position, agent):
        tmp = deepcopy(self.positions_left_to_visit[agent])
        if position in self.positions_left_to_visit[agent]:
            del tmp[tmp.index(position)]

        self.positions_left_to_visit[agent] = tmp

        found_all = False
        if len(self.positions_left_to_visit[agent]) == 0:
            found_all = True
        return found_all

    # ================= #
    # ==== Is Goal ==== #
    # ================= #
    def is_goal(self, position):
        found_goal = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                found_goal = True
                break

        return found_goal

    # =================== #
    # ==== Get Image ==== #
    # =================== #
    def get_image(self):
        return self.maze.to_rgb()

    # ==================== #
    # ==== Build Maze ==== #
    # ==================== #
    def build_maze(self, width, height, max_num_shapes, max_size, seed=None):
        self.maze = RoombaMazeGenerator2(width, height, max_num_shapes, max_size, seed)
        self.maze.objects.agent.positions = [self.start_idx[0]] * self.num_agents
        self.maze.objects.goal.positions = self.goal_idx

        impassable_array = self.maze.to_impassable()
        start = self.start_idx[0]
        goal = self.goal_idx[0]
        actions = dijkstra_solver(impassable_array, self.motions, start, goal)

        while True:
            if not actions:
                self.maze = RoombaMazeGenerator2(width, height, max_num_shapes, max_size, seed)
            else:
                break

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        # self.maze.objects.agent.positions = self.start_idx
        self.maze.objects.agent.positions = [self.start_idx[0]] * self.num_agents
        self.maze.objects.goal.positions = self.goal_idx

    # ============================ #
    # ==== Agents Communicate ==== #
    # ============================ #
    def agents_communicate(self, agent):
        current_position = self.maze.objects.agent.positions[agent]
        found_potential_agents, potential_pos_list = self.lidars[agent].scan_for_objects(self.maze.maze_layout, current_position)

        agents_to_send_info = []
        if found_potential_agents:
            for pos in potential_pos_list:
                if pos != current_position and pos in self.maze.objects.agent.positions:
                    agent_index = self.maze.objects.agent.positions.tolist().index(pos)
                    agents_to_send_info.append(agent_index)

        return agents_to_send_info

    # ================================ #
    # ==== Get LiDAR Scan Figures ==== #
    # ================================ #
    def get_lidar_scan_figures(self):
        if self.maze.objects:
            for object in self.maze.objects:
                if object.name != 'obstacle':
                    for pos in object.positions:
                        x = pos[1]
                        y = pos[0]

                        free2 = Object('free2', 0, color.free, False, [pos])

                        # self.lidar.scan(self.maze.maze_layout, free2)
                        self.lidars[0].scan(self.maze.maze_layout, free2[0])

                        print(object.name, 'x:', x, ' y:', y)

                        # degree_list = [index for index, info in enumerate(self.lidar.angle_dist_coord_pt)]
                        # degree_list = [index for index, info in enumerate(self.lidar.small_angle_dist_list)]
                        degree_list = [index for index, info in enumerate(self.lidars[0].small_angle_dist_list)]
                        # distance_list = deepcopy(self.lidar.angle_dist_coord_pt)
                        # distance_list = deepcopy(self.lidar.small_angle_dist_list)
                        distance_list = deepcopy(self.lidars[0].small_angle_dist_list)
                        y_pos = np.arange(len(degree_list))

                        fig, ax = plt.subplots()
                        ax.bar(y_pos, distance_list, align='center', alpha=0.5)
                        ax.set(xlabel='Degree Angle', ylabel='Distance', title='Position x: {0} y: {1}'.format(x, y))

                        if not os.path.exists('./saved_files/lidar_scans/'):
                            os.makedirs('./saved_files/lidar_scans/')

                        fig.savefig('./saved_files/lidar_scans/x_{0}_y_{1}'.format(x, y))
                        plt.close()
        else:
            print('WARNING: Maze has not been built yet. Unable to get LIDAR scans')



class Lidar:
    def __init__(self, position_tup, env, distance=5, angle=0, test_mode=False):
        """
        The Constructor
        """
        self.x = position_tup[0]
        self.y = position_tup[1]
        self.distance = distance
        self.angle = 0
        self.lidar = (0, 0)
        self.coord_pts_data = []
        self.angle_dist_coord_pt = []
        self.small_angle_dist_list = []
        # self.small_degrees = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]  # 16
        # self.small_degrees = [0, 45, 90, 135, 180, 225, 270, 330]  # 8
        self.small_degrees = [0, 90, 180, 270]  # 4

        self.degree_step_size = 1
        if self.small_degrees and len(self.small_degrees) == 4:
            self.degree_step_size = 90

        self.mode = test_mode
        self.env = env
        # self.obstacles_map = self.env.get_object_map()
        self.rotation = ['NORTH', 90]  # This means its facing NORTH on the map aka forward is up

    def scan(self, map_layout, vehicle_position):
        x = vehicle_position[1]
        y = vehicle_position[0]
        self.angle_dist_coord_pt = [0] * 360

        dist = []
        # cur_object = self.env.maze.generator.free
        cur_object = 0
        for i in range(0, 360):
            dist = [x, y]
            for j in range(1, 40):
                # if i == 179:
                    # v = 1
            # x1 = int(self.distance * cos(((i + self.rotation[1]) * 3.14) / 180) + x)
            # y1 = int(self.distance * sin(((i - self.rotation[1]) * 3.14) / 180) + y)
                x1 = int(x + self.distance * cos((i * 3.14) / 180) * j / 40)
                y1 = int(y + self.distance * sin((i * 3.14) / 180) * j / 40)

                size = self.env.maze.size
                if x1 > 0 and x1 < size[1] and y1 > 0 and y1 < size[0]:
                    cur_object = map_layout[y1][x1]

                if x1 <= 0 or x1 >= size[1] or y1 <= 0 or y1 >= size[0]:
                    break
                # elif cur_object.name.upper() == 'OBSTACLE':
                #     break
                elif cur_object == 1:
                    break
                else:
                    dist = [x1, y1]

            if len(self.coord_pts_data) < 360:
                self.coord_pts_data.append(dist)
            else:
                self.coord_pts_data[i] = dist

            if x > 0 and x < size[1] and y > 0 and y < size[0]:
                vect = Vector()
                vect.cal_vector((x, y), self.coord_pts_data[i])
                scan_dist = vect.get_magnitude()
                if len(self.angle_dist_coord_pt) < 360:
                    # self.angle_dist_coord_pt.append([i, scan_dist])
                    self.angle_dist_coord_pt.append(scan_dist)
                else:
                    self.angle_dist_coord_pt[i] = scan_dist

        self.small_angle_dist_list = [self.angle_dist_coord_pt[degree_angle] for degree_angle in self.small_degrees]


    def scan_for_objects(self, map_layout, vehicle_position):
        found_potential_person = False
        potential_pos_list = []

        x = vehicle_position[1]
        y = vehicle_position[0]

        dist = []
        # cur_object = self.env.maze.generator.free
        cur_object = 0
        size = self.env.maze.size
        for i in range(0, 360, self.degree_step_size):
            dist = [x, y]
            for j in range(1, 40):
                x1 = int(x + self.distance * cos((i * 3.14) / 180) * j / 40)
                y1 = int(y + self.distance * sin((i * 3.14) / 180) * j / 40)

                if x1 > 0 and x1 < size[1] and y1 > 0 and y1 < size[0]:
                    cur_object = map_layout[y1][x1]

                if x1 <= 0 or x1 >= size[1] or y1 <= 0 or y1 >= size[0]:
                    break
                elif cur_object in (1, 2) and [y1, x1] != vehicle_position:
                    break
                else:
                    dist = [x1, y1]

            if x > 0 and x < size[1] and y > 0 and y < size[0]:
                vect = Vector()
                vect.cal_vector((x, y), dist)
                scan_dist = vect.get_magnitude()
                if scan_dist < self.distance:
                    found_potential_person = True
                    potential_pos_list.append([dist[1], dist[0]])

        return found_potential_person, potential_pos_list

    def show(self):
        for dot_matrix_pt in self.coord_pts_data:
            current_vector = Vector()
            current_vector.cal_vector((self.x, self.y), dot_matrix_pt)

            if current_vector.get_magnitude() < 190:
                print(self.coord_pts_data, (255, 0, 0))

    def move(self, position): #, rotation):
        self.x = position[0]
        self.y = position[1]

        # if rotation[0] == 'NORTH' and self.rotation[0] != 'NORTH':
        #     self.rotation[0] = 'NORTH'
        #     self.rotation[1] -= 90
        # elif rotation[0] == 'SOUTH' and self.rotation[0] != 'NORTH':
        #     self.rotation[0] = 'NORTH'
        #     self.rotation[1] -= 90
        # elif rotation[0] == 'NORTH' and self.rotation[0] != 'NORTH':
        #     self.rotation[0] = 'NORTH'
        #     self.rotation[1] -= 90
        # elif rotation[0] == 'NORTH' and self.rotation[0] != 'NORTH':
        #     self.rotation[0] = 'NORTH'
        #     self.rotation[1] -= 90






class Vector(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def cal_vector(self, p1, p2):
        self.x = p2[0] - p1[0]
        self.y = p2[1] - p1[1]
        return self.x, self.y

    def get_magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)





class DQN:
    def __init__(self, env, state_size):
        self.env = env
        # Model Hyperparameters
        self.state_size = 360  # This number could be 720 we are providing both (angle, distance)
        self.action_size = env.action_space.n
        self.learning_rate = 10e-5  # This is alpha

        # Training Hyperparameters
        self.total_episodes = 5000
        self.max_episode_steps = 3000
        self.batch_size = 32

        # Q Target Hyperparameter
        self.max_tau = 0.125

        # Exploration Hyperparameter
        self.epsilon = 2.0
        self.min_epsilon = 0.1
        self.decay = 0.999

        # Q Learning Hyperparameter
        self.gamma = 0.95

        # Memory Replay Hyperparamets
        self.memory_size = 4000
        self.memory = deque(maxlen=self.memory_size)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.loss = None
        self.steps = 0

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(96, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, state_1, done):
        self.memory.append((state, action, reward, state_1, done))

    def act(self, state):
        self.steps += 1

        if self.steps % 100 == 0:
            self.epsilon *= self.decay
            self.epsilon = max(self.min_epsilon, self.epsilon)

        selected_action = 0
        if random.random() < self.epsilon:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = np.argmax(self.model.predict(state)[0])

        return selected_action

    def replay(self, batch_size):
        mini_batch_samples = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch_samples:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + q_future * self.gamma

            self.model.fit(state, target, batch_size=len(mini_batch_samples), epochs=1, verbose=0)
            self.loss = self.model.evaluate(state, target, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.max_tau + target_weights[i] * (1 - self.max_tau)

        self.target_model.set_weights(target_weights)

    def save(self, name):
        self.model.save(name, overwrite=True)
        self.model.save_weights(name.replace('.h5', '-weights.h5'), overwrite=True)

    def load(self, name, use_weights=None):

        if not use_weights:
            del self.model
            self.model = keras.models.load_model(name)
        else:
            self.model.load_weights(name)
            # self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))


# https://keon.io/deep-q-learning/
# https://github.com/keon/deep-q-learning/blob/master/ddqn.py
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        # self.learning_rate = 10e-5
        # self.epsilon_decay = 0.99985
        # self.learning_rate = 10e-5
        self.batch_size = 32
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.count = 0
        self.is_finished = False

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(LSTM(24, return_sequences=False, input_shape=(1, self.state_size)))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.learning_rate))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))



        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(48, activation='relu'))
        # model.add(Dense(96, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        self.count += 1
        # self.count % 100 == 0  and
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










def get_random_env(use_defaults=False, seed=None, gen_maze_width_and_width=8, num_maze_shapes=3, num_agents=1, training_type='escape'):
    if not use_defaults:
        gen_maze_width_and_width = random.randrange(8, 101)
        num_maze_shapes = int(max(gen_maze_width_and_width, gen_maze_width_and_width) / 2 - 1)

    if not seed and use_defaults:
        seed = 1

    env = RoombaEnv2(training_type=training_type,
                     num_of_agents=num_agents,
                     width=gen_maze_width_and_width,
                     height=gen_maze_width_and_width,
                     max_num_shapes=num_maze_shapes,
                     max_size=8,
                     seed=seed)

    return env


def train(training_type, num_agents, num_of_epochs, path_to_saved_data, path_to_trained_model=None, seed=None, network_type='dqn', render=False):
    use_defaults = False
    if seed:
        use_defaults = True

    num_of_agents = num_agents
    env = get_random_env(use_defaults, seed, num_agents=num_of_agents, training_type=training_type)

    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'

    # network_obj = None
    network_objs = None
    results_dict = dict()
    episode_steps = 500

    if network_type == 'dqn':
        state_size = 5
        print('Training: DQN')
        # dqn = DQNAgent(360, env.action_space.n)
        # dqn = DQNAgent(16, env.action_space.n)
        # dqn = DQNAgent(17, env.action_space.n)
        # dqn = DQNAgent(9, env.action_space.n)
        # dqn = DQNAgent(5, env.action_space.n)
        dqns = [DQNAgent(state_size, env.action_space.n)] * num_of_agents
        states = [[]] * num_of_agents
        cumlative_episode_rewards = [0] * num_of_agents
        total_steps_takens = [0] * num_of_agents


        if path_to_trained_model:
            # dqn.load(path_to_trained_model)
            dqns[0].load(path_to_trained_model)

        for epoch in tqdm(range(num_of_epochs)):
            if not seed:
                env = get_random_env()

            for agent_index, dqn in enumerate(dqns):
                state = env.reset(agent_index)
                # state = np.reshape(state, [1, state_size])
                state = np.reshape(state, [1, 1, state_size])
                dqn.is_finished = False
                states[agent_index] = state
                cumlative_episode_rewards[agent_index] = 0
                total_steps_takens[agent_index] = 0


            # state = np.reshape(state, [1, 360])
            # state = np.reshape(state, [1, 16])
            # state = np.reshape(state, [1, 17])
            # state = np.reshape(state, [1, 9])
            # state = np.reshape(state, [1, 5])
            # state = np.reshape(state, [1, 1, 16])
            # print(state.shape)
            # cumlative_episode_reward = 0
            # for step in tqdm(range(episode_steps)):
            # step = 0
            # total_steps_taken = 0
            # while True:
            finished_count = 0
            for step in range(episode_steps):
                # step += 1
                if render:
                    env.render()

                for agent_num, dqn in enumerate(dqns):
                    if dqn.is_finished:
                        pass

                    if len(dqns) > 1:
                        comm_agents_list = env.agents_communicate(agent_num)
                        if comm_agents_list:
                            for comm_agent_index in comm_agents_list:
                                if len(dqns[comm_agent_index].memory) > 0:
                                    env.seen_lidar_scans[agent_num] = env.seen_lidar_scans[agent_num] + list(set(env.seen_lidar_scans[comm_agent_index] - set(env.seen_lidar_scans[agent_num])))
                                    comm_state, action, reward, comm_next_state, done = dqns[comm_agent_index].memory[-1]
                                    dqn.remember(comm_state, action, reward, comm_next_state, done)
                                    if len(dqn.memory) > dqn.batch_size:
                                        dqn.replay(dqn.batch_size)

                    # action = dqn.act(state)
                    action = dqn.act(states[agent_num])

                    # next_state, reward, done, info = env.step(action)
                    next_state, reward, done, info = env.step(action, agent_num)

                    # next_state = np.reshape(next_state, [1, 360])
                    # next_state = np.reshape(next_state, [1, 16])
                    # next_state = np.reshape(next_state, [1, 17])
                    # next_state = np.reshape(next_state, [1, 9])
                    # next_state = np.reshape(next_state, [1, state_size])
                    next_state = np.reshape(next_state, [1, 1, state_size])
                    # next_state = np.reshape(state, [1, 1, 16])
                    dqn.remember(state, action, reward, next_state, done)

                    state = next_state
                    states[agent_num] = state

                    cumlative_episode_rewards[agent_num] += reward
                    total_steps_takens[agent_num] = step
                    if done:
                        dqn.update_target_model()
                        print("agent: {} epoch: {}/{} episode: {}/{}, score: {}, epsilon_val: {:.2}, invalid: {}, seen: {}, new: {}"
                              .format(agent_num, epoch, num_of_epochs, step, 'UNLIMITED', cumlative_episode_rewards[agent_num], dqn.epsilon,
                                      env.num_of_invalid_moves, env.num_of_seen_moves, env.num_of_new_moves))
                        finished_count += 1

                    if len(dqn.memory) > dqn.batch_size:
                        dqn.replay(dqn.batch_size)

                if finished_count == len(dqns):
                    break

            for num_agent, dqn in enumerate(dqns):
                if epoch % 100 == 0:
                    model_file_path = path_to_saved_data + "new_maze-dqn-LRate-{0}-{1}epIter_Agent{2}.h5".format(dqn.learning_rate, episode_steps, num_agent).replace('-0.','-')
                    print('Saving Model: ', model_file_path)
                    dqn.save(model_file_path)

                if num_agent not in results_dict:
                    results_dict[num_agent] = dict()

                results_dict[num_agent][epoch] = [total_steps_takens[num_agent], cumlative_episode_rewards[num_agent],
                                       env.num_of_invalid_moves, env.num_of_seen_moves, env.num_of_new_moves]

                # results_dict[epoch] = {num_agent: [total_steps_takens[num_agent], cumlative_episode_rewards[num_agent],
                #                        env.num_of_invalid_moves, env.num_of_seen_moves, env.num_of_new_moves]}

            if not seed:
                env.close()

        # Save final Training weights and results
        for num_agent, dqn in enumerate(dqns):
            model_file_path = path_to_saved_data + "FINAL_maze-dqn-{}epochs_Agent{}.h5".format(num_of_epochs, num_agent)
            dqn.save(model_file_path)

            with open(path_to_saved_data + 'FINAL_results_dict-{}_Agent{}.txt'.format(num_of_epochs, num_agent), 'w') as output:
                output.write(str(results_dict))
        #
        # network_obj = dqn
        network_objs = dqns
        #
    else:
        print('unsupported algorithm test')

    return network_objs


def run_in_env(env, dqn_obj):
    env.viewer = None
    # env = Monitor(env, directory='./', force=True)
    state = env.reset()
    # state = np.reshape(state, [1, 360])
    # state = np.reshape(state, [1, 16])
    # state = np.reshape(state, [1, 17])
    # state = np.reshape(state, [1, 9])
    state = np.reshape(state, [1, 5])
    # state = np.reshape(state, [1, 1, 16])

    num_states_taken = 0
    total_reward = 0

    for i in range(500):
        num_states_taken += 1
        env.render()

        # Select action
        action = dqn_obj.act(state)

        next_state, reward, done, info = env.step(action)
        # state = np.reshape(next_state, [1, 360])
        # state = np.reshape(next_state, [1, 16])
        # state = np.reshape(next_state, [1, 17])
        # state = np.reshape(next_state, [1, 9])
        state = np.reshape(next_state, [1, 5])
        # state = np.reshape(next_state, [1, 1, 16])

        total_reward += reward

        if done:
            break

        time.sleep(0.25)

    env.close()

    return total_reward, num_states_taken, env.num_of_invalid_moves, env.num_of_seen_moves, env.num_of_new_moves


def create_graphs(xdata, ydata, label_info_list, save_path):
    xlabel_str = label_info_list[0]
    ylabel_str = label_info_list[1]
    title_str = label_info_list[2]

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set(xlabel=xlabel_str, ylabel=ylabel_str, title=title_str)
    ax.grid()
    fig.savefig(save_path)


def plot_results(path_to_results, figure_base_name):
    base_dir = os.path.split(path_to_results)[0] + '/'

    with open(path_to_results, 'r') as input:
        results_dict_str = input.readlines()[0]

    if results_dict_str != '':
        results_dict = ast.literal_eval(results_dict_str)

        for agent_results in results_dict:

            lists = sorted(results_dict[agent_results].items())  # sorted by key, return a list of tuples

            epoch_num, epoch_info = zip(*lists)  # unpack a list of pairs into two tuples
            steps = [val[0] for val in list(epoch_info)]
            ep_rewards = [val[1] for val in list(epoch_info)]
            invalid_count = [val[2] for val in list(epoch_info)]
            seen_count = [val[3] for val in list(epoch_info)]
            new_count = [val[4] for val in list(epoch_info)]

            # Graph steps taken
            labels_list = ['Epoch Number', 'Number Of Steps Taken', 'Number of Steps Taken Within an Epoch']
            create_graphs(epoch_num, steps, labels_list, base_dir + figure_base_name + 'StepsTaken.png')

            # Graph rewards
            labels_list = ['Epoch Number', 'Cumulative Reward', 'Cumulative Reward Within an Epoch']
            create_graphs(epoch_num, ep_rewards, labels_list, base_dir + figure_base_name + 'CumReward.png')

            # Graph Invalid Moves
            labels_list = ['Epoch Number', 'Num Of Invalid Actions Attempted', 'Number of Invalid Actions Attempted Within An Epoch']
            create_graphs(epoch_num, invalid_count, labels_list, base_dir + figure_base_name + 'NumInvalid.png')

            # Graph Seen Scans
            labels_list = ['Epoch Number', 'Num Of Previously Seen LIDAR Scans', 'Num Of Previously Seen LIDAR Scans Within An Epoch']
            create_graphs(epoch_num, seen_count, labels_list, base_dir + figure_base_name + 'NumSeenScans.png')

            # Graph New Scans
            labels_list = ['Epoch Number', 'Num Of New LIDAR Scans', 'Num Of New LIDAR Scans Within An Epoch']
            create_graphs(epoch_num, new_count, labels_list, base_dir + figure_base_name + 'NumNewScans.png')


if __name__ == '__main__':

    # plot_results('./saved_files/new_results_dict-100.txt')
    # path_to_trained_model = './saved_files/training_run_8x8sweep/FINAL_maze-dqn-1000epochs_Agent0-weights.h5'
    path_to_trained_model = None
    need_training = True
    # need_training = False
    # network_type = 'actor_critic'
    network_type = 'dqn'
    current_DQN = None
    num_of_training_episodes = 1000

    path_to_saved_data = './saved_files/training_run_4x4_sweep/'

    if not os.path.exists(path_to_saved_data):
        os.makedirs(path_to_saved_data)

    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'

    env_8x8 = RoombaEnv2(training_type='sweep', num_of_agents=1, width=8, height=8, max_num_shapes=3, max_size=8, seed=1)
    print('Created 8x8 Map Environment')

    # env_16x16 = RoombaEnv2(num_of_agents=1, width=16, height=16, max_num_shapes=8, max_size=8, seed=2)
    # print('Created 16x16 Map Environment')
    #
    # env_32x32 = RoombaEnv2(num_of_agents=1, width=32, height=32, max_num_shapes=15, max_size=8, seed=3)
    # print('Created 32x32 Map Environment')
    #
    # env_64x64 = RoombaEnv2(num_of_agents=1, width=64, height=64, max_num_shapes=50, max_size=8, seed=1)
    # print('Created 64x64 Map Environment')

    if need_training:
        # Train the DQN
        will_render = True
        # will_render = False
        num_agents = 1
        seed = 1
        current_DQN_list = train('sweep', num_agents, num_of_training_episodes, path_to_saved_data, path_to_trained_model, seed, network_type, render=will_render)
    else:
        if network_type == 'dqn':
            use_weights = True
            # use_weights = False

            # current_DQN = DQN(env_8x8, env_8x8.reset())
            # current_DQN = DQNAgent(360, env_8x8.action_space.n)
            # current_DQN = DQNAgent(16, env_8x8.action_space.n)
            # current_DQN = DQNAgent(17, env_8x8.action_space.n)
            # current_DQN = DQNAgent(9, env_8x8.action_space.n)
            current_DQN = DQNAgent(5, env_8x8.action_space.n)
            current_DQN.load(path_to_trained_model)

    if network_type == 'dqn':
        current_DQN.epsilon = 0.0
        print('finished Training')

        # Run the trained model in different environments
        env_8x8_reward, steps_taken_8x8, invalid_moves, seen_moves, new_moves = run_in_env(env_8x8, current_DQN)
        print(env_8x8_reward, '  STEPS: ', steps_taken_8x8, ' INVALID: ', invalid_moves, 'SEEN: ', seen_moves, 'NEW: ', new_moves)

        path_to_results = path_to_saved_data + 'FINAL_results_dict-{}_Agent0.txt'.format(num_of_training_episodes)
        fig_base_name = 'TrainingDQN_{}Epoch_Agent0'.format(num_of_training_episodes)
        plot_results(path_to_results, fig_base_name)

        # current_DQN.env = env_16x16
        # env_16x16_reward, steps_taken_16x16, invalid_moves, seen_moves, new_moves = run_in_env(env_16x16, current_DQN)
        #
        # current_DQN.env = env_32x32
        # env_32x32_reward, steps_taken_32x32, invalid_moves_32x32, seen_moves_32x32, new_moves_32x32 = run_in_env(env_32x32, current_DQN)
        #
        # current_DQN.env = env_64x64
        # env_64x64_reward, steps_taken_64x64, invalid_moves_64x64, seen_moves_64x64, new_moves_64x64 = run_in_env(env_64x64, current_DQN)


# Change input to 16 pts rather than 360 degrees = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]
# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py