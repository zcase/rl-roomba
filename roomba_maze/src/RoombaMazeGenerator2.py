from copy import deepcopy
import numpy as np
from mazelab2 import BaseMaze
from mazelab2 import Object
from mazelab2 import DeepMindColor as color
from mazelab2.generators import random_shape_maze, random_maze


# =============================== #
# ==== Roomba Maze Generator ==== #
# =============================== #
class RoombaMazeGenerator2(BaseMaze):

    # ============== #
    # ==== Init ==== #
    # ============== #
    def __init__(self, maze_type='random_shape', width=50, height=50, max_num_shapes=50, max_size=8, seed=None):
        # super(RoombaMazeGenerator, self).__init__()

        self.maze_layout = None
        if maze_type == 'random_shape':
            self.maze_layout = random_shape_maze(width, height, max_num_shapes, max_size, allow_overlap=False, shape=None, seed=seed)
        elif maze_type == 'random_maze':
            self.maze_layout = random_maze(width=width, height=height, complexity=1.0, density=1.0, seed=seed)

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
        # agent2 = Object('agent', 4, (255, 255, 204), False, [])
        # agent = Object('agent', 2, color.agent, True, [])
        goal = Object('goal', 3, color.goal, False, [])
        # lidar = Object('lidar', 4, (255, 255, 204), False, [])
        return free, obstacle, agent, goal
