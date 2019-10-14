import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import time
from tqdm import tqdm
from src.RoombaEnv2 import RoombaEnv2
from src.RoombaMazeGenerator2 import RoombaMazeGenerator2
from mazelab2.solvers import dijkstra_solver
import gym
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow as tf
import random
from copy import deepcopy
from keras.layers import *
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam
import itertools


from keras import backend as K

from src.DQN import DQNAgent
import ast


def get_random_env(maze_info, use_defaults=False):
    if not use_defaults:
        gen_maze_width_and_width = random.randrange(8, 101)
        num_maze_shapes = int(max(gen_maze_width_and_width, gen_maze_width_and_width) / 2 - 1)
        maze_info_dict['num_shapes'] = num_maze_shapes

    env = RoombaEnv2(maze_type=maze_info['maze_type'],
                     training_type=maze_info['mode'],
                     num_of_agents=maze_info['num_of_agents'],
                     width=maze_info['width'],
                     height=maze_info['height'],
                     max_num_shapes=maze_info['num_shapes'],
                     max_size=8,
                     seed=maze_info['seed'])

    return env


def communicate_with_agents(env, agent_num, dqns, step, recv_or_send='recv'):
    comm_to_agents_list = env.agents_communicate(agent_num)
    if comm_to_agents_list:
        for comm_agent_index in comm_to_agents_list:
            if recv_or_send.lower() == 'recv':
                if len(dqns[comm_agent_index].memory) > 0 and not dqns[comm_agent_index].is_finished:
                    for comm_state, action, reward, comm_next_state, done in list(itertools.islice(dqns[comm_agent_index].memory, len(dqns[comm_agent_index].memory) - step,
                                             len(dqns[comm_agent_index].memory))):

                        if done:
                            for i in range(5):
                                dqns[agent_num].remember(comm_state, action, reward, comm_next_state, done)
                        else:
                            dqns[agent_num].remember(comm_state, action, reward, comm_next_state, done)

                    if len(dqns[agent_num].memory) > dqns[agent_num].batch_size:
                        dqns[agent_num].replay(dqns[agent_num].batch_size)

            elif recv_or_send.lower() == 'send':
                for comm_state, action, reward, comm_next_state, done in list(
                        itertools.islice(dqns[agent_num].memory, len(dqns[agent_num].memory) - step,
                                         len(dqns[agent_num].memory))):

                    if done:
                        for i in range(5):
                            dqns[comm_agent_index].remember(comm_state, action, reward, comm_next_state, done)
                    else:
                        dqns[comm_agent_index].remember(comm_state, action, reward, comm_next_state, done)

                if len(dqns[comm_agent_index].memory) > dqns[comm_agent_index].batch_size:
                    dqns[comm_agent_index].replay(dqns[comm_agent_index].batch_size)

    return dqns


def train(maze_info_dict, num_of_epochs, path_to_saved_data, path_to_trained_model=None, network_type='dqn', render=False):
    use_defaults = False
    seed = maze_info_dict['seed']
    num_of_agents = maze_info_dict['num_of_agents']
    training_type = maze_info_dict['mode']
    mazetype = maze_info_dict['maze_type']
    maze_w = maze_info_dict['width']
    maze_h = maze_info_dict['height']

    if seed:
        use_defaults = True

    env = get_random_env(maze_info_dict, use_defaults)

    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'

    # network_obj = None
    network_objs = None
    results_dict = dict()
    episode_steps = 500

    if network_type == 'dqn':
        # state_size = 5
        # state_size = 361
        # state_size = 9 * 3
        # state_size = 9
        state_size = 5
        print('Training: DQN')
        dqns = []
        for i in range(num_of_agents):
            dqns.append(deepcopy(DQNAgent(state_size, env.action_space.n)))

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
                state = np.reshape(state, [1, state_size])
                dqn.is_finished = False
                states[agent_index] = state
                cumlative_episode_rewards[agent_index] = 0
                total_steps_takens[agent_index] = 0

            finished_agents = [False] * num_of_agents
            for step in range(episode_steps):
                # if render:
                #     env.render()

                for agent_num, dqn in enumerate(dqns):
                    if render:
                        env.render()
                    if not dqn.is_finished:
                        if len(dqns) > 1:
                            dqns = communicate_with_agents(env, agent_num, dqns, step)

                        action = dqn.act(states[agent_num])

                        next_state, reward, done, info = env.step(action, agent_num)
                        next_state = np.reshape(next_state, [1, state_size])

                        dqn.remember(state, action, reward, next_state, done)

                        # if reward == -10:
                        #     for i in range(10):
                        #         dqn.remember(state, action, reward, next_state, done)

                        state = next_state
                        states[agent_num] = state

                        # if len(dqn.memory) > dqn.batch_size:
                        #     dqn.replay(dqn.batch_size)

                        cumlative_episode_rewards[agent_num] += reward
                        total_steps_takens[agent_num] = step
                        if done:
                            # for i in range(10):
                            #     dqn.remember(state, action, reward, next_state, done)
                            dqn.update_target_model()
                            print("agent: {} epoch: {}/{} episode: {}/{}, score: {}, epsilon_val: {:.2}, invalid: {}, seen: {}, new: {}, loss: {}"
                                  .format(agent_num, epoch, num_of_epochs, step, 'UNLIMITED', cumlative_episode_rewards[agent_num], dqn.epsilon,
                                          env.num_of_invalid_moves[agent_num], env.num_of_seen_moves[agent_num], env.num_of_new_moves[agent_num], dqn.loss))
                            finished_agents[agent_num] = True
                            env.maze.objects.agent.positions[agent_num] = [0, 0]
                            if len(dqns) > 1:
                                dqns = communicate_with_agents(env, agent_num, dqns, step, 'send')
                            dqn.is_finished = True

                if all(finished_agents):
                    break

            for num_agent, dqn in enumerate(dqns):
                if len(dqn.memory) > dqn.batch_size:
                    dqn.replay(dqn.batch_size)

            for num_agent, dqn in enumerate(dqns):
                if epoch % 100 == 0:
                    model_file_path = path_to_saved_data + "new_maze-dqn-LRate-{0}-{1}epIter_Agent{2}.h5".format(dqn.learning_rate, episode_steps, num_agent).replace('-0.','-')
                    print('Saving Model: ', model_file_path)
                    dqn.save(model_file_path)

                if num_agent not in results_dict:
                    results_dict[num_agent] = dict()

                results_dict[num_agent][epoch] = [total_steps_takens[num_agent], cumlative_episode_rewards[num_agent],
                                       env.num_of_invalid_moves[num_agent], env.num_of_seen_moves[num_agent], env.num_of_new_moves[num_agent], dqn.loss]

                if epoch % 10 == 0:
                    with open(path_to_saved_data + 'results_dict_Agent{}.txt'.format(num_agent),'w') as output:
                        output.write(str(results_dict))

            # For random Map Generation
            if not seed:
                env.close()

            if epoch % 100 == 0:
                for agent_num in range(num_of_agents):
                    env.create_movement_heatmap(agent_num, path_to_saved_data, epoch)

        # Save final Training weights and results
        for num_agent, dqn in enumerate(dqns):
            model_file_path = path_to_saved_data + "FINAL_maze-dqn-{}epochs_Agent{}.h5".format(num_of_epochs, num_agent)
            dqn.save(model_file_path)

            with open(path_to_saved_data + 'FINAL_results_dict-{}_Agent{}.txt'.format(num_of_epochs, num_agent), 'w') as output:
                output.write(str(results_dict))

        network_objs = dqns
    else:
        print('unsupported algorithm test')

    return network_objs


def run_in_env(env, dqn_objs, state_size, action_size):
    env.viewer = None
    states = [[]] * len(dqn_objs)
    total_rewards = [0] * len(dqn_objs)
    num_states_taken = [0] * len(dqn_objs)

    for agent_index, dqn in enumerate(dqn_objs):
        state = env.reset(agent_index)
        state = np.reshape(state, [1, state_size])
        dqn.is_finished = False
        states[agent_index] = state
        total_rewards[agent_index] = 0
        num_states_taken[agent_index] = 0
    # env = Monitor(env, directory='./', force=True)
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    # num_states_taken = 0
    # total_reward = 0

    finished_agents = [False] * len(dqn_objs)
    for i in range(500):
        for agent_num, dqn in enumerate(dqn_objs):
            env.render()
            if not dqn.is_finished:
                num_states_taken[agent_num] += 1


                # Select action
                action = dqn.act(state)

                next_state, reward, done, info = env.step(action, agent_num)

                state = np.reshape(next_state, [1, state_size])

                total_rewards[agent_num] += reward

                if done:
                    finished_agents[agent_num] = True
                    env.maze.objects.agent.positions[agent_num] = [0, 0]
                    dqn.is_finished = True
                    break

                time.sleep(0.25)

        if all(finished_agents):
            break

    env.close()

    return total_rewards, num_states_taken, env.num_of_invalid_moves, env.num_of_seen_moves, env.num_of_new_moves


def create_graphs(xdata, ydata, label_info_list, save_path):
    xlabel_str = label_info_list[0]
    ylabel_str = label_info_list[1]
    title_str = label_info_list[2]

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set(xlabel=xlabel_str, ylabel=ylabel_str, title=title_str)
    ax.grid()
    fig.savefig(save_path)
    plt.close(fig)


def plot_results(path_to_results, figure_base_name, agent_num=0):
    base_dir = os.path.split(path_to_results)[0] + '/'

    with open(path_to_results, 'r') as input:
        results_dict_str = input.readlines()[0]

    if results_dict_str != '':
        results_dict = ast.literal_eval(results_dict_str)

        # for agent_results in results_dict:

        lists = sorted(results_dict[agent_num].items())  # sorted by key, return a list of tuples

        epoch_num, epoch_info = zip(*lists)  # unpack a list of pairs into two tuples
        steps = [val[0] for val in list(epoch_info)]
        ep_rewards = [val[1] for val in list(epoch_info)]
        invalid_count = [val[2] for val in list(epoch_info)]
        seen_count = [val[3] for val in list(epoch_info)]
        new_count = [val[4] for val in list(epoch_info)]

        # Graph steps taken
        labels_list0 = ['Epoch Number', 'Number Of Steps Taken', 'Number of Steps Taken Within an Epoch: Agent_{}'.format(agent_num)]
        create_graphs(epoch_num, steps, labels_list0, base_dir + figure_base_name + 'StepsTaken.png')

        # Graph rewards
        labels_list1 = ['Epoch Number', 'Cumulative Reward', 'Cumulative Reward Within an Epoch: Agent_{}'.format(agent_num)]
        create_graphs(epoch_num, ep_rewards, labels_list1, base_dir + figure_base_name + 'CumReward.png')

        # Graph Invalid Moves
        labels_list2 = ['Epoch Number', 'Num Of Invalid Actions Attempted', 'Number of Invalid Actions Attempted Within An Epoch: Agent_{}'.format(agent_num)]
        create_graphs(epoch_num, invalid_count, labels_list2, base_dir + figure_base_name + 'NumInvalid.png')

        # Graph Seen Scans
        labels_list3 = ['Epoch Number', 'Num Of Previously Seen LIDAR Scans', 'Num Of Previously Seen LIDAR Scans Within An Epoch: Agent_{}'.format(agent_num)]
        create_graphs(epoch_num, seen_count, labels_list3, base_dir + figure_base_name + 'NumSeenScans.png')

        # Graph New Scans
        labels_list4 = ['Epoch Number', 'Num Of New LIDAR Scans', 'Num Of New LIDAR Scans Within An Epoch: Agent_{}'.format(agent_num)]
        create_graphs(epoch_num, new_count, labels_list4, base_dir + figure_base_name + 'NumNewScans.png')

        # Create a combined Graph
        fig2, (ax0, ax1, ax2) = plt.subplots(nrows=3)
        ax0.plot(epoch_num, steps)
        ax0.set_title(labels_list0[2])
        # ax0.set( ylabel=labels_list0[1])

        ax1.plot(epoch_num, ep_rewards)
        ax1.set_title(labels_list1[2])
        # ax1.set(ylabel=labels_list1[1])

        ax2.plot(epoch_num, invalid_count)
        ax2.set_title(labels_list2[2])
        ax2.set(xlabel=labels_list2[0])

        fig2.tight_layout()
        fig2.savefig(base_dir + figure_base_name + '_MASTER_Graph_1.png')
        plt.close(fig2)

        fig3, (ax3, ax4) = plt.subplots(nrows=2)
        ax3.plot(epoch_num, seen_count)
        ax3.set_title(labels_list3[2])
        # ax3.set(ylabel=labels_list3[1])

        ax4.plot(epoch_num, new_count)
        ax4.set_title(labels_list4[2])
        ax4.set(xlabel=labels_list4[0])

        fig3.tight_layout()
        fig3.savefig(base_dir + figure_base_name + '_MASTER_Graph_2.png')
        plt.close(fig3)



if __name__ == '__main__':

    path_to_trained_model = None
    need_training = False
    # need_training = True

    network_type = 'dqn'

    current_DQN = None
    num_of_training_episodes = 1000
    # state_size = 361
    # state_size = 9 * 3
    # state_size = 9
    state_size = 5
    action_size = 4

    # Maze info
    maze_info_dict = {
        'maze_type': 'random_shape',
        # 'mode': 'sweep',
        'mode': 'escape',
        # 'width': 7,
        'width': 5,
        # 'width': 4,
        # 'height': 11,
        # 'height': 8,
        'height': 7,
        # 'height': 4,
        # 'num_shapes': 3,
        # 'num_shapes': 4,
        'num_shapes': 3,
        # 'num_shapes': 0,
        'seed': 6,
        # 'num_of_agents': 2
        'num_of_agents': 1
    }

    # Rendering
    will_render = True
    # will_render = False

    # path_to_saved_data = './saved_files/training_run/'
    # path_to_saved_data = './saved_files/training_run_8x8_escape_MultiAgent/'
    path_to_saved_data = './saved_files/training_run_5x7_school/'

    if not os.path.exists(path_to_saved_data):
        os.makedirs(path_to_saved_data)

    if not path_to_saved_data.endswith('/'):
        path_to_saved_data += '/'

    # Use for testing
    dqns = []

    # Train the DQN
    if need_training:
        dqns = train(maze_info_dict,
                     num_of_training_episodes,
                     path_to_saved_data,
                     path_to_trained_model,
                     network_type,
                     render=will_render)

        print('finished Training')
        need_training = False
    # Test the Model
    else:
        if network_type == 'dqn':
            for i in range(maze_info_dict['num_of_agents']):
                dqns.append(deepcopy(DQNAgent(state_size, action_size)))
                path_to_trained_model = path_to_saved_data + 'FINAL_maze-dqn-1000epochs_Agent{0}-weights.h5'.format(i)
                dqns[i].load(path_to_trained_model)
                # dqns[i].epsilon = 0.1
                dqns[i].epsilon = 0.0

    if network_type == 'dqn' and not need_training:
        print('Testing.......')

        env = RoombaEnv2(maze_type=maze_info_dict['maze_type'],
                         training_type=maze_info_dict['mode'],
                         num_of_agents=maze_info_dict['num_of_agents'],
                         width=maze_info_dict['width'],
                         height=maze_info_dict['height'],
                         max_num_shapes=maze_info_dict['num_shapes'],
                         max_size=8,
                         seed=maze_info_dict['seed'])

        # # Run the trained model in different environments
        env_8x8_reward, steps_taken_8x8, invalid_moves, seen_moves, new_moves = run_in_env(env, dqns, state_size, action_size)
        print(env_8x8_reward, '  STEPS: ', steps_taken_8x8, ' INVALID: ', invalid_moves, 'SEEN: ', seen_moves, 'NEW: ', new_moves)

        for agent_num in range(len(dqns)):
            path_to_results = path_to_saved_data + 'FINAL_results_dict-{}_Agent{}.txt'.format(num_of_training_episodes, agent_num)
            fig_base_name = 'TrainingDQN_{}Epoch_Agent{}'.format(num_of_training_episodes, agent_num)
            plot_results(path_to_results, fig_base_name, agent_num)

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