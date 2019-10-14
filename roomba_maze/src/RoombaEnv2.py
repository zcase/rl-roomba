import os
import numpy as np
from mazelab2 import BaseEnv
from mazelab2 import VonNeumannMotion
from mazelab2 import Object
from mazelab2 import DeepMindColor as color
import gym
from gym.spaces import Box
from gym.spaces import Discrete
from src.RoombaMazeGenerator2 import RoombaMazeGenerator2
from mazelab2.solvers import dijkstra_solver
from .LidarSim import Lidar
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import deque


class RoombaEnv2(BaseEnv):

    # ============== #
    # ==== Init ==== #
    # ============== #
    def __init__(self, maze_type='random_shape', training_type='escape', num_of_agents=1, width=50, height=50, max_num_shapes=50, max_size=8, seed=None):
        super().__init__()

        self.start_idx = None
        self.goal_idx = None
        self.maze = None
        self.motions = VonNeumannMotion()
        self.num_agents = num_of_agents
        self.agent_move_history = [deepcopy(deque(maxlen=3))] * num_of_agents

        if maze_type == 'random_shape':
            self.env_id = 'RandomShapeMaze-v0'
            self.start_idx = [[1, 1]]
            self.goal_idx = [[height - 2, width - 2]]
            self.build_random_shape_maze(maze_type, width, height, max_num_shapes, max_size, seed=seed)
        elif maze_type == 'random_maze':
            self.start_idx = [[1, 1]]
            self.goal_idx = [[height - 1, width - 1]]
            self.env_id = 'RandomMaze-v0'
            self.build_random_maze(width, height, seed)

        self.training_type = training_type
        print('Training to ', self.training_type, ' an area!')
        self.maze_height = height
        self.maze_width = width

        # self.lidar = Lidar(self.start_idx[0], self)
        self.lidars = [Lidar(self.start_idx[0], self)] * num_of_agents
        # self.slam = RMHC_SLAM(LaserModel(), self.maze.maze_width, 5)
        # self.mapbytes = bytearray(self.maze.maze_width * self.maze.maze_height)
        self.seen_lidar_scans = deepcopy([[]] * num_of_agents)
        self.num_of_invalid_moves = deepcopy([0] * num_of_agents)
        self.num_of_seen_moves = deepcopy([0] * num_of_agents)
        self.num_of_new_moves = deepcopy([0] * num_of_agents)

        self.positions_counter = []
        tmp_maze = deepcopy(self.maze.maze_layout)
        for i in range(self.num_agents):
            self.positions_counter.append(deepcopy(tmp_maze))

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
            self.maze.objects.agent.positions[agent] = new_position
            self.lidars[agent].move([new_position[1], new_position[0]])
        else:
            new_position = current_position

        self.positions_counter[agent][new_position[0]][new_position[1]] += 1

        self.lidars[agent].scan(self.maze.maze_layout, self.maze.objects.agent.positions[agent])

        found_new_spot = False
        if self.lidars[agent].angle_dist_coord_pt not in self.seen_lidar_scans[agent]:
        # if self.lidars[agent].small_angle_dist_list not in self.seen_lidar_scans[agent]:
            found_new_spot = True
            self.seen_lidar_scans[agent].append(self.lidars[agent].angle_dist_coord_pt)
            # self.seen_lidar_scans[agent].append(self.lidars[agent].small_angle_dist_list)

        # Get Reward and Check if agent is done
        reward, done = self.get_reward(new_position, agent, found_new_spot, valid)

        # observation = deepcopy(self.lidars[agent].small_angle_dist_list)
        observation = deepcopy(self.lidars[agent].angle_dist_coord_pt)
        if found_new_spot:
            observation.append(1)  # Has seen a new spot
        else:
            observation.append(-1)  # Has seen spot before

        self.agent_move_history.append(observation)
        # cur_obs = self.agent_move_history[agent][2] + self.agent_move_history[agent][1] + self.agent_move_history[agent][0]
        return observation, reward, done, {}
        # return cur_obs, reward, done, {}

    # =============== #
    # ==== Reset ==== #
    # =============== #
    def reset(self, agent=0):
        self.num_of_invalid_moves = [0] * self.num_agents
        self.num_of_seen_moves = [0] * self.num_agents
        self.num_of_new_moves = [0] * self.num_agents
        self.agent_move_history = [deepcopy(deque(maxlen=3))] * self.num_agents

        for i in range(self.agent_move_history[agent].maxlen):
            self.agent_move_history[agent].append([0] * 9)

        self.positions_counter = []
        tmp_maze = deepcopy(self.maze.maze_layout)
        for i in range(self.num_agents):
            self.positions_counter.append(deepcopy(tmp_maze))

        self.maze.objects.agent.positions = [self.start_idx[0]] * self.num_agents
        self.maze.objects.goal.positions = self.goal_idx

        self.positions_counter[agent][self.start_idx[0][0]][self.start_idx[0][1]] += 1

        self.lidars[agent].scan(self.maze.maze_layout, self.maze.objects.agent.positions[agent])
        self.seen_lidar_scans[agent] = []
        # self.seen_lidar_scans[agent].append(deepcopy(self.lidars[agent].small_angle_dist_list))
        self.seen_lidar_scans[agent].append(deepcopy(self.lidars[agent].angle_dist_coord_pt))
        self.num_of_new_moves[agent] += 1

        # observation = deepcopy(self.lidars[agent].small_angle_dist_list)
        observation = deepcopy(self.lidars[agent].angle_dist_coord_pt)
        observation.append(1)
        self.agent_move_history[agent].append(observation)
        # cur_obs = self.agent_move_history[agent][2] + self.agent_move_history[agent][1] + self.agent_move_history[agent][0]
        return observation
        # return cur_obs

    # ==================== #
    # ==== Get Reward ==== #
    # ==================== #
    def get_reward(self, new_position, agent, found_new_spot, valid):
        reward = 0
        done = False

        # Reward structure from getting from Point A to Point B
        if self.training_type == 'escape':
            if self.is_goal(new_position):
                self.num_of_new_moves[agent] += 1
                reward = +10  # Reward used for DQN to find path in maze
                done = True
            elif not valid:
                self.num_of_invalid_moves[agent] += 1
                reward = -10
                # reward = -200
            elif not found_new_spot:
                reward = -0.3 * self.positions_counter[agent][new_position[0]][new_position[1]]  # Reward used for DQN to find path in maze
                # reward = -0.5
                self.num_of_seen_moves[agent] += 1

            elif found_new_spot:
                # reward = +0.3  # Reward used for DQN to find path in maze
                # reward = +1  # Reward used for DQN to find path in maze
                reward = +1.5  # Reward used for DQN to find path in maze
                self.num_of_new_moves[agent] += 1


        # Reward structure for searching all of gridword
        elif self.training_type == 'sweep':
            if self.searched_complete_grid(agent):
                if found_new_spot:
                    reward = +0.95
                    self.num_of_new_moves[agent] += 1
                reward += +10
                done = True
            elif not valid:
                self.num_of_invalid_moves[agent] += 1
                reward = -10
                # reward = -100 * (1 - self.num_of_invalid_moves[agent])
            elif not found_new_spot:
                reward = -0.3 * self.positions_counter[agent][new_position[0]][new_position[1]]
                # reward = -100 * self.positions_counter[agent][new_position[0]][new_position[1]]
                self.num_of_seen_moves[agent] += 1
            elif found_new_spot:
                reward = +5
                self.num_of_new_moves[agent] += 1

        return reward, done

    # ================== #
    # ==== Is Valid ==== #
    # ================== #
    def is_valid(self, position):
        # Check and make sure agent isn't off grid in x or y postion
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]

        # Make sure the position an agent is going to is free to move to.
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        no_other_agent_in_pos = True
        if self.num_agents > 1:
            if position in self.maze.objects.agent.positions:
                no_other_agent_in_pos = False

        # Make sure position passes ALL 3 validation criteria
        is_valid = nonnegative and within_edge and passable and no_other_agent_in_pos

        return is_valid

    # ================================ #
    # ==== Searched Complete Grid ==== #
    # ================================ #
    def searched_complete_grid(self, agent):
        found_all = False
        result = np.where(self.positions_counter[agent] == 0)
        coords_left_to_visit = list(zip(result[0], result[1]))

        if not coords_left_to_visit:
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
    def build_random_shape_maze(self, maze_type, width, height, max_num_shapes, max_size, seed=None):
        self.maze = RoombaMazeGenerator2(maze_type, width, height, max_num_shapes, max_size, seed)
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

    def build_random_maze(self, width, height, seed):
        self.maze = RoombaMazeGenerator2(maze_type='random_maze', width=width, height=height, seed=seed)
        self.maze.objects.agent.positions = [self.start_idx[0]] * self.num_agents
        self.maze.objects.goal.positions = self.goal_idx

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
                    agent_index = self.maze.objects.agent.positions.index(pos)
                    agents_to_send_info.append(agent_index)

        return agents_to_send_info

    # ================================ #
    # ==== Get LiDAR Scan Figures ==== #
    # ================================ #
    def get_lidar_scan_figures(self):
        if self.maze.objects:
            for object in self.maze.objects:
                if object.name not in ('agent', 'obstacle'):
                    for pos in object.positions:
                        x = pos[1]
                        y = pos[0]

                        if not isinstance(pos, list):
                            pos = pos.tolist()

                        free2 = Object('free2', 0, color.free, False, pos)

                        # self.lidar.scan(self.maze.maze_layout, free2)
                        self.lidars[0].scan(self.maze.maze_layout, free2.positions)

                        print(object.name, 'x:', x, ' y:', y)

                        degree_list = [index for index, info in enumerate(self.lidars[0].angle_dist_coord_pt)]
                        # degree_list = [index for index, info in enumerate(self.lidar.small_angle_dist_list)]
                        # degree_list = [index for index, info in enumerate(self.lidars[0].small_angle_dist_list)]
                        distance_list = deepcopy(self.lidars[0].angle_dist_coord_pt)
                        # distance_list = deepcopy(self.lidar.small_angle_dist_list)
                        # distance_list = deepcopy(self.lidars[0].small_angle_dist_list)
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

    # ================================= #
    # ==== Create Movement Heatmap ==== #
    # ================================= #
    def create_movement_heatmap(self, agent, output_dir='./', epoch_num=None):
        fig, ax = plt.subplots()
        surf = ax.imshow(self.positions_counter[agent])

        height, width = self.positions_counter[agent].shape

        for i in range(height):
            for j in range(width):
                ax.text(j, i, self.positions_counter[agent][i][j], ha='center', va='center', color='w')

        heatmap_label_divider = make_axes_locatable(ax)
        plot_axis = heatmap_label_divider.append_axes("right", size="5%", pad="5%")
        heatmap_bar = plt.colorbar(surf, ticks=[0, 1], cax=plot_axis)
        heatmap_bar.ax.set_yticklabels(['Visited Less', 'Visited More'])
        heatmap_bar.ax.invert_yaxis()

        if not output_dir.endswith('/'):
            output_dir += '/'

        filename = 'heatmap_agent_{0}'.format(agent)
        if epoch_num:
            filename += '_epoch_{0}'.format(str(epoch_num))

        filename += '.png'

        fig.savefig(output_dir + filename)

