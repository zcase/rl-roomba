# To give credit where credit is do, this code is based off of the works found at the following sites:
#   https://github.com/raykking/DQN_lidar
#   This is part of the code that was provided by paper [3] Dynamic Path Planning of Unknown Environment Based on Deep Reinforcement Learning
# I have taken the information from the site and modified it into work for my representation below.

from math import sqrt, cos, sin, atan2, pi, ceil
from src.RoombaUtils import Vector
from bresenham import bresenham


class Lidar:

    # ============== #
    # ==== Init ==== #
    # ============== #
    def __init__(self, position_tup, env, distance=5, angle=0, test_mode=False):
        """
        The Constructor
        """
        self.x = position_tup[0]
        self.y = position_tup[1]
        self.distance = distance
        self.coord_pts_data = []
        self.angle_dist_coord_pt = []
        self.small_angle_dist_list = []

        # self.small_degrees = [0] * 360
        # self.small_degrees = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]  # 16
        # self.small_degrees = [0, 45, 90, 135, 180, 225, 271, 330]  # 8
        self.small_degrees = [0, 90, 180, 271]  # 4

        self.degree_step_size = ceil(360.0 / len(self.small_degrees))

        print(360.0/len(self.small_degrees))

        self.mode = test_mode
        self.env = env
        # self.obstacles_map = self.env.get_object_map()
        self.rotation = ['NORTH', 90]  # This means its facing NORTH on the map aka forward is up

    # ============== #
    # ==== Scan ==== #
    # ============== #
    def scan(self, map_layout, vehicle_position):
        x = vehicle_position[1]
        y = vehicle_position[0]
        self.angle_dist_coord_pt = [0] * int(360 / self.degree_step_size)

        dist = []
        # cur_object = self.env.maze.generator.free
        cur_object = 0
        # for i in range(0, 360):
        for i in range(0, 360, self.degree_step_size):
            if i == 270:
                i = 271

            i_val = ceil(360.0 / self.degree_step_size)
            degree_index = int(i / self.degree_step_size)

            dist = [x, y]
            for j in range(1, 40):
                x1 = int(x + self.distance * cos((i * 3.14) / 180) * j / 40)
                y1 = int(y + self.distance * sin((i * 3.14) / 180) * j / 40)

                size = self.env.maze.size
                if x1 > 0 and x1 < size[1] and y1 > 0 and y1 < size[0]:
                    cur_object = map_layout[y1][x1]

                if x1 <= 0 or x1 >= size[1] or y1 <= 0 or y1 >= size[0]:
                    break
                # elif cur_object.name.upper() == 'OBSTACLE':
                #     break
                elif cur_object == 1 or ([y1, x1] in self.env.maze.objects.agent.positions and [y1, x1] != vehicle_position):
                    break
                else:
                    dist = [x1, y1]

            # if len(self.coord_pts_data) < 360:
            if len(self.coord_pts_data) < i_val:
                self.coord_pts_data.append(dist)
            else:
                self.coord_pts_data[degree_index] = dist
                # self.coord_pts_data[i] = dist

            if x > 0 and x < size[1] and y > 0 and y < size[0]:
                vect = Vector()
                vect.cal_vector((x, y), self.coord_pts_data[degree_index])
                # vect.cal_vector((x, y), self.coord_pts_data[i])
                scan_dist = vect.get_magnitude()
                # if len(self.angle_dist_coord_pt) < 360:
                if len(self.angle_dist_coord_pt) < i_val:
                    self.angle_dist_coord_pt.append(scan_dist)
                else:
                    self.angle_dist_coord_pt[degree_index] = scan_dist
                    # self.angle_dist_coord_pt[i] = scan_dist

            if i == 271:
                i = 270

        # self.small_angle_dist_list = [self.angle_dist_coord_pt[degree_angle] for degree_angle in self.small_degrees]
        # self.angle_dist_coord_pt = self.small_angle_dist_list

    # ========================== #
    # ==== Scan For Objects ==== #
    # ========================== #
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
            # dist = [x, y]
            for j in range(1, 40):
                x1 = int(x + self.distance * cos((i * 3.14) / 180) * j / 40)
                y1 = int(y + self.distance * sin((i * 3.14) / 180) * j / 40)

                if x1 > 0 and x1 < size[1] and y1 > 0 and y1 < size[0]:
                    cur_object = map_layout[y1][x1]

                if x1 <= 0 or x1 >= size[1] or y1 <= 0 or y1 >= size[0]:
                    break
                elif cur_object == 1 or ([y1, x1] in self.env.maze.objects.agent.positions and [y1, x1] != vehicle_position):
                    dist = [x1, y1]
                # else:
                #     dist = [x1, y1]

            if dist and x > 0 and x < size[1] and y > 0 and y < size[0] and [dist[1], dist[0]] != vehicle_position:
                vect = Vector()
                vect.cal_vector((x, y), dist)
                scan_dist = vect.get_magnitude()
                if scan_dist < self.distance:
                    found_potential_person = True

                    for points in list(bresenham(x, y, dist[0], dist[1]))[1:]:
                        if map_layout[points[1]][points[0]] == 1:
                            dist = [points[0], points[1]]
                            break
                    if [dist[1], dist[0]] not in potential_pos_list:
                        potential_pos_list.append([dist[1], dist[0]])

        # if len(potential_pos_list) >= len(self.small_degrees):
        # potential_pos_list = [potential_pos_list[degree_angle] for degree_angle in self.small_degrees]
        # if vehicle_position in potential_pos_list:
        #     potential_pos_list = list(filter(lambda pos: pos != vehicle_position, potential_pos_list))

        if potential_pos_list:
            found_potential_person = True

        return found_potential_person, potential_pos_list


    def move(self, position): #, rotation):
        self.x = position[0]
        self.y = position[1]
