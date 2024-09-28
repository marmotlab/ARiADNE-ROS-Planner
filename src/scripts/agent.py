import time
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as colors
import parameter

from utils import *
# from parameter import *
from node_manager import NodeManager


class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        # location and map
        self.location = None
        self.map_info = None

        # map related parameters
        self.cell_size = parameter.CELL_SIZE
        self.node_resolution = parameter.NODE_RESOLUTION
        self.updating_map_size = parameter.UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = NodeManager()

        # graph
        self.node_coords, self.utility, self.guidepost = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        # rarefied graph
        self.key_node_coords, self.key_utility, self.key_guidepost = None, None, None
        self.key_current_index, self.key_adjacent_matrix, self.key_neighbor_indices = None, None, None

    def update_map(self, map_info):
        self.map_info = map_info

    def update_updating_map(self, location):
        # the updating map is the part of the global map that maybe affected by new measurements
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited()

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[
                                     0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[
                                     1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.location = self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        t1 = time.time()
        self.node_manager.get_rarefied_graph(self.location, self.map_info)
        t2 = time.time()
        # print("graph rarefaction", t2 - t1)
        # self.node_coords, self.utility, self.guidepost, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
        #     self.update_observation()
        t1 = time.time()
        self.key_node_coords, self.key_utility, self.key_guidepost, self.key_adjacent_matrix, self.key_current_index, self.key_neighbor_indices = \
            self.update_key_node_observation()
        t2 = time.time()
        # print("update key node graph", t2 - t1)

    def update_observation(self):
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                assert index is not None
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def update_key_node_observation(self):
        all_key_node_coords = []
        for key_node_coords in self.node_manager.key_node_dict.keys():
            all_key_node_coords.append(np.array(key_node_coords))
        all_key_node_coords = np.array(all_key_node_coords).reshape(-1, 2)
        utility = []
        guidepost = []

        n_nodes = all_key_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_key_node_coords[:, 0] + all_key_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_key_node_coords):
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            utility.append(node.utility)
            guidepost.append(node.visited)
            for neighbor in node.neighbor_set:
                neighbor = np.array([neighbor[0], neighbor[1]])
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        guidepost = np.array(guidepost)

        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        return all_key_node_coords, utility, guidepost, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self, robot_location):

        node_coords = deepcopy(self.key_node_coords)
        node_utility = self.key_utility.reshape(-1, 1)
        node_guidepost = self.key_guidepost.reshape(-1, 1)
        current_index = self.key_current_index
        edge_mask = self.key_adjacent_matrix
        current_edge = self.key_neighbor_indices

        node_coords[current_index] = robot_location

        current_node_coords = robot_location
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                     axis=-1) / parameter.UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (parameter.UTILITY_RANGE * 3.14 // parameter.FRONTIER_CELL_SIZE)
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0).to(self.device)
        k_size = current_edge.size()[-1]
        current_edge = current_edge.unsqueeze(-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        return [node_inputs, None, edge_mask, current_index, current_edge, edge_padding_mask]

    def get_next_observation(self, next_node_index, observation):
        node_inputs, _, edge_mask, curren_index, _, _ = observation
        next_edge = torch.argwhere(edge_mask[0, next_node_index] == 0).flatten()
        next_in_edge = torch.argwhere(next_edge == next_node_index).item()
        curren_in_edge = torch.argwhere(next_edge == curren_index.item()).item()
        k_size = next_edge.size()[-1]
        next_edge = next_edge.unsqueeze(-1).unsqueeze(0)
        next_node_index = torch.tensor([next_node_index]).reshape(1, 1, 1).to(self.device)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, next_in_edge] = 1
        edge_padding_mask[0, 0, curren_in_edge] = 1
        return node_inputs, None, edge_mask, next_node_index, next_edge, edge_padding_mask

    def select_next_waypoint(self, observation, greedy=True):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.key_node_coords[next_node_index]
        # print("available next positions:", self.key_node_coords[current_edge[0].numpy()].reshape(-1, 2))

        return next_position, next_node_index

    def plot_env(self, step, robot_location):
        # quite slow, only use it to debug

        # plt.switch_backend('TKAgg')
        plt.ion()
        plt.clf()

        plt.subplot(1, 2, 1)
        nodes = get_cell_position_from_coords(self.key_node_coords, self.map_info).reshape(-1, 2)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(robot_location, self.map_info)
        # plt.imshow(self.map_info.map, cmap='gray')
        plt.imshow(self.map_info.map + 1.1, cmap='gray_r', norm=colors.LogNorm())
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.key_utility, zorder=2)
        for node, utility in zip(nodes, self.key_utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        for coords in self.key_node_coords:
            node = self.node_manager.key_node_dict[(coords[0], coords[1])]
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.subplot(1, 2, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(robot_location, self.map_info)
        plt.imshow(self.map_info.map + 1.1, cmap='gray_r', norm=colors.LogNorm())
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for node, utility in zip(nodes, self.utility):
            plt.text(node[0], node[1], str(utility), zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        for coords in self.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                         (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.pause(1e-3)

        plt.savefig('{}/{}_samples.png'.format(f'gifs', step), dpi=150)
        # plt.close()
