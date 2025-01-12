import time
import rospy
import heapq
from copy import deepcopy

from torch.fx.proxy import orig_method_name

import parameter
import math
import numpy as np
from utils import *
import quads


class NodeManager:
    def __init__(self, start=np.array([0,0])):
        # use quad tree to index nodes in the original graph
        self.nodes_dict = quads.QuadTree((0,0), 1000, 1000)
        self.add_node_to_dict(start, [], None)
        # use dictionary to index nodes in the rarefied graph
        self.key_node_dict = {}
        # use dictionary to index cluster center node
        self.cluster_center_node_dict = {}
        # quad tree is initialized at
        self.start = start

        # dstar-lite related
        self.new_nodes = set()
        self.removed_nodes = set()
        self.updated_edges = set()

        # save path to nearest frontier
        self.path_to_nearest_frontier = None
        self.dist_to_nearest_frontier = 1e8

        self.last = start


    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        key = (coords[0], coords[1])
        node = Node(coords, frontiers, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)

        return node

    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_set:
            if neighbor_coords[0] == node.coords[0] and neighbor_coords[1] == node.coords[1]:
                continue
            neighbor_node = self.nodes_dict.find(neighbor_coords).data
            neighbor_node.neighbor_set.remove((node.coords[0], node.coords[1]))

        self.nodes_dict.remove(node.coords.tolist())

    def check_valid_node(self, robot_location, map_info):
        checking_boundary = get_quad_tree_box(robot_location, 2 * (parameter.SENSOR_RANGE + parameter.NODE_RESOLUTION))
        node_to_check = self.nodes_dict.within_bb(checking_boundary)
        for node in node_to_check:
            node = node.data
            coords = node.coords
            if not is_free(coords, map_info):
                self.removed_nodes.add(node)
                self.remove_node_from_dict(node)

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        # only get the nodes in the updating map
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        global_frontiers = get_frontier_in_map(map_info)

        # update nodes
        t1 = time.time()
        all_node_list = []
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
                self.new_nodes.add(node)
            else:
                node = node.data
                # utilities of nodes out of this range are not affected by new measurements
                if np.linalg.norm(node.coords - robot_location) > 2 * parameter.SENSOR_RANGE:
                    pass
                # nodes with zero utility is likely not affected
                elif np.linalg.norm(
                        node.coords - robot_location) > parameter.THR_GRAPH_HARD_UPDATE and node.utility == 0:
                    pass
                elif np.linalg.norm(node.coords - robot_location) < parameter.THR_GRAPH_HARD_UPDATE and node.utility > 0:
                    node.initialize_observable_frontiers(frontiers, updating_map_info)
                else:
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info, global_frontiers)
            all_node_list.append(node)
        t2 = time.time()
        # print("update nodes", t2 - t1)

        for node in all_node_list:
            updated_edges = set()
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    parameter.SENSOR_RANGE + parameter.NODE_RESOLUTION * 3):
                updated_edges = node.update_neighbor_nodes(updating_map_info, self.nodes_dict)
            elif np.linalg.norm(node.coords - robot_location) < parameter.THR_GRAPH_HARD_UPDATE:
                updated_edges = node.update_neighbor_nodes(updating_map_info, self.nodes_dict, hard_update=True)
            self.updated_edges = self.updated_edges.union(updated_edges)

        t3 = time.time()
        # print("update edges", t3 - t2)

        # remove nodes unconnected to the origin
        self.remove_unconnected_nodes(self.start)
        t4 = time.time()

        redundant_nodes = self.new_nodes & self.removed_nodes
        self.new_nodes = self.new_nodes - redundant_nodes
        self.removed_nodes = self.removed_nodes - redundant_nodes

        current_node = self.nodes_dict.find(robot_location.tolist())
        if current_node is not None:
            self.last = robot_location
            return robot_location
        else:
            rospy.loginfo("The current node should be removed.")
            nearest_node = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data
            self.last = nearest_node.coords
            return nearest_node.coords

    def remove_unconnected_nodes(self, origin):
        all_nodes_coords = set()
        for node in self.nodes_dict.__iter__():
            node = node.data
            all_nodes_coords.add((node.coords[0], node.coords[1]))

        current_node = self.nodes_dict.find(origin.tolist()).data
        all_connected_nodes = set()
        all_connected_nodes.add((current_node.coords[0], current_node.coords[1]))
        nodes_to_expand = deepcopy(current_node.neighbor_set)

        while len(nodes_to_expand) > 0:
            new_nodes_to_expand = set()
            for node in nodes_to_expand:
                neighbors = self.nodes_dict.find(node).data.neighbor_set
                for neighbor in neighbors:
                    if neighbor not in all_nodes_coords:
                        pass
                    elif neighbor in all_connected_nodes:
                        pass
                    else:
                        new_nodes_to_expand.add(neighbor)
                all_connected_nodes.add(node)
            nodes_to_expand = new_nodes_to_expand

        not_connected_nodes = all_nodes_coords.difference(all_connected_nodes)

        for node in not_connected_nodes:
            node = self.nodes_dict.find(node).data
            self.removed_nodes.add(node)
            self.remove_node_from_dict(node)

    def get_rarefied_graph(self, robot_location, map_info):
        self.dist_to_nearest_frontier = 1e8

        t1 = time.time()
        # get all nodes with non-zero utility
        non_zero_utility_nodes = set()
        for node in self.nodes_dict.__iter__():
            if node.data.utility > 0:
                non_zero_utility_nodes.add(node.data)

        # we are going to cluster them
        clustered_non_zero_utility_nodes = set()

        # first find the current node and add it to the rarefied graph
        self.key_node_dict = {}
        current_node = self.nodes_dict.find(robot_location.tolist()).data
        key_current_node = KeyNode(current_node.coords, current_node.utility, current_node.visited)
        self.key_node_dict[(key_current_node.coords[0], key_current_node.coords[1])] = key_current_node

        # The first cluster center is the current node
        cluster_boundary = get_quad_tree_box(current_node.coords, 1 * parameter.SENSOR_RANGE)
        nodes_in_cluster = self.nodes_dict.within_bb(cluster_boundary)
        nodes_in_cluster = [node.data for node in nodes_in_cluster]

        # we find paths to all non-zero-utility nodes in the first cluster and add them to the rarefied graph
        for node in nodes_in_cluster:
            if node.utility > 0:

                if (node.coords[0], node.coords[1]) in self.key_node_dict.keys():
                    non_zero_utility_nodes.remove(node)
                    continue

                path, dist = self.a_star(current_node.coords, node.coords)

                if dist == 1e8:
                    continue
                if dist < self.dist_to_nearest_frontier:
                    self.dist_to_nearest_frontier = dist
                    self.path_to_nearest_frontier = path

                non_zero_utility_nodes.remove(node)

                ref = (robot_location[0], robot_location[1])
                ref_node = self.key_node_dict[ref]
                first_node = True
                for i, coords in enumerate(path):
                    if first_node:
                        # the first node on path should not be too close to the current node, unless it is out of sight or with non-zero utility
                        if coords not in self.nodes_dict.find(ref).data.neighbor_set and (
                                check_collision(np.array(ref), np.array(coords), map_info) or np.linalg.norm(
                            np.array(coords) - np.array(ref)) > (
                                        parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION)):
                            assert i >= 1, print(self.nodes_dict.find(ref).data.neighbor_set, path, dist)

                            key_node_coords = path[i - 1]
                            if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                                key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                            else:
                                node = self.nodes_dict.find(key_node_coords).data
                                key_node = KeyNode(node.coords, node.utility, node.visited)
                                self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node
                            key_node.add_neighbor_coords(ref)
                            ref_node.add_neighbor_coords(key_node_coords)
                            first_node = False

                            key_node_coords = coords
                            if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                                pass
                            else:
                                node = self.nodes_dict.find(key_node_coords).data
                                key_node = KeyNode(node.coords, node.utility, node.visited)
                                self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node

                        elif self.nodes_dict.find(coords).data.utility > 0:
                            key_node_coords = coords
                            if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                                key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                            else:
                                node = self.nodes_dict.find(key_node_coords).data
                                key_node = KeyNode(node.coords, node.utility, node.visited)
                                self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node
                            key_node.add_neighbor_coords(ref)
                            ref_node.add_neighbor_coords(key_node_coords)
                            first_node = False

                    else:
                        key_node_coords = coords
                        if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                            pass
                            # key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                        else:
                            node = self.nodes_dict.find(key_node_coords).data
                            key_node = KeyNode(node.coords, node.utility, node.visited)
                            self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node

        # check existing cluster centers
        cluster_center_nodes_to_delete = []
        for center_node, _ in self.cluster_center_node_dict.values():
            if center_node.utility == 0 or center_node in nodes_in_cluster:
                cluster_center_nodes_to_delete.append(center_node)
                continue
            if self.nodes_dict.find((center_node.coords[0], center_node.coords[1])) is None:
                cluster_center_nodes_to_delete.append(center_node)
                continue
            self.update_clustered_nodes(center_node, clustered_non_zero_utility_nodes)

        for center_node in cluster_center_nodes_to_delete:
            del self.cluster_center_node_dict[(center_node.coords[0], center_node.coords[1])]

        # find other cluster centers through cluster range and nodes' connectivity
        for center_node in non_zero_utility_nodes:
            if center_node in clustered_non_zero_utility_nodes:
                continue

            if parameter.ENABLE_DSTARLITE:
                dstar = DStarLite(self.nodes_dict, (robot_location[0], robot_location[1]), (center_node.coords[0], center_node.coords[1]))
            else:
                dstar = None

            self.cluster_center_node_dict[(center_node.coords[0], center_node.coords[1])] = [center_node, dstar]
            self.update_clustered_nodes(center_node, clustered_non_zero_utility_nodes)

        t2 = time.time()
        # print("find cluster center", t2 - t1)

        t_d = 0
        # for other cluster we only calculate the path to the cluster center
        # print("number of cluster:", len(self.cluster_center_node_dict))
        find_one_path = False
        i = 0
        for center_node, dstar in self.cluster_center_node_dict.values():
            i += 1
            # center_node_coords = self.nodes_dict.find(center_node.coords.tolist()).data.coords
            t11 = time.time()
            if dstar is not None:
                start = (robot_location[0], robot_location[1])
                if not dstar.first_run:
                    dstar.update_graph(new_start=start, new_nodes=self.new_nodes, deleted_nodes=self.removed_nodes, updated_edges=self.updated_edges)
                else:
                    dstar.first_run = False
                path, dist = dstar.get_shortest_path()
                if dist == 1e8:
                    rospy.logdebug("Dstarlite exceed time limit")  # it is OK because most likely this node is too far, we can keep search it
            else:
                path, dist = self.a_star(robot_location, center_node.coords)

            t12 = time.time()
            t_d += t12 - t11

            if dist == 1e8:
                if find_one_path:
                    continue
                elif i < len(self.cluster_center_node_dict):
                    continue
                else:
                    if parameter.ENABLE_DSTARLITE:
                        path, dist = self.a_star(robot_location, center_node.coords)
                    else:
                        continue

            find_one_path = True

            if dist < self.dist_to_nearest_frontier:
                self.dist_to_nearest_frontier = dist
                self.path_to_nearest_frontier = path

            ref = (robot_location[0], robot_location[1])
            ref_node = self.key_node_dict[ref]
            first_node = True
            for i, coords in enumerate(path):
                if first_node:
                    if coords not in self.nodes_dict.find(ref).data.neighbor_set and (
                            check_collision(np.array(ref), np.array(coords), map_info) or np.linalg.norm(
                        np.array(coords) - np.array(ref)) > (parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION)):
                        assert i >= 1, print(self.nodes_dict.find(ref).data.neighbor_set, path, dist)
                        key_node_coords = path[i - 1]
                        if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                            key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                        else:
                            node = self.nodes_dict.find(key_node_coords).data
                            key_node = KeyNode(node.coords, node.utility, node.visited)
                            self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node
                        key_node.add_neighbor_coords(ref)
                        ref_node.add_neighbor_coords(key_node_coords)
                        first_node = False

                        key_node_coords = coords
                        if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                            pass
                        else:
                            node = self.nodes_dict.find(key_node_coords).data
                            key_node = KeyNode(node.coords, node.utility, node.visited)
                            self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node

                    elif self.nodes_dict.find(coords).data.utility > 0:
                        key_node_coords = coords
                        if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                            key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                        else:
                            node = self.nodes_dict.find(key_node_coords).data
                            key_node = KeyNode(node.coords, node.utility, node.visited)
                            self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node
                        key_node.add_neighbor_coords(ref)
                        ref_node.add_neighbor_coords(key_node_coords)
                        first_node = False

                else:
                    key_node_coords = coords
                    if (key_node_coords[0], key_node_coords[1]) in self.key_node_dict.keys():
                        key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                    else:
                        node = self.nodes_dict.find(key_node_coords).data
                        key_node = KeyNode(node.coords, node.utility, node.visited)
                        self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node

        # maybe we miss some nodes
        for node in non_zero_utility_nodes:
            coords = (node.coords[0], node.coords[1])
            if coords not in self.key_node_dict.keys():
                key_node = KeyNode(node.coords, node.utility, node.visited)
                self.key_node_dict[(coords[0], coords[1])] = key_node

        # find neighbors for key nodes
        for node_coords in self.key_node_dict.keys():
            key_node = self.key_node_dict[node_coords]
            node = self.nodes_dict.find(node_coords).data
            for neighbor_coords in node.neighbor_set:
                if neighbor_coords in self.key_node_dict.keys():
                    key_node.add_neighbor_coords(neighbor_coords)

        self.new_nodes = set()
        self.removed_nodes = set()
        self.updated_edges = set()
        t3 = time.time()
        # assert len(self.key_node_dict) > 1
        # print("get sparse graph", t3 - t2)

    def update_clustered_nodes(self, center_node, clustered_non_zero_utility_nodes):
        nodes_to_check = set()

        cluster_boundary = get_quad_tree_box(center_node.coords, 2 * parameter.CLUSTER_RANGE)
        nodes_in_cluster = self.nodes_dict.within_bb(cluster_boundary)
        non_zero_utility_nodes_in_cluster = set()
        for node in nodes_in_cluster:
            if node.data.utility > 0:
                non_zero_utility_nodes_in_cluster.add(node.data)

        for neighbor_coords in center_node.neighbor_set:
            if neighbor_coords == (center_node.coords[0], center_node.coords[1]):
                continue
            neighbor_node = self.nodes_dict.find(neighbor_coords).data
            if neighbor_node in non_zero_utility_nodes_in_cluster:
                clustered_non_zero_utility_nodes.add(neighbor_node)
                nodes_to_check.add(neighbor_node)

        while len(nodes_to_check) > 0:
            new_nodes_to_check = set()
            for node in nodes_to_check:
                for coords in node.neighbor_set:
                    if coords == (node.coords[0], node.coords[1]):
                        continue
                    neighbor_node = self.nodes_dict.find(coords).data
                    if neighbor_node in clustered_non_zero_utility_nodes:
                        continue
                    if neighbor_node not in non_zero_utility_nodes_in_cluster:
                        continue
                    clustered_non_zero_utility_nodes.add(neighbor_node)
                    for neighbor_coords in node.neighbor_set:
                        if neighbor_coords == (node.coords[0], node.coords[1]):
                            continue
                        neighbor_node = self.nodes_dict.find(neighbor_coords).data
                        if neighbor_node in non_zero_utility_nodes_in_cluster and neighbor_node not in clustered_non_zero_utility_nodes:
                            new_nodes_to_check.add(neighbor_node)
            nodes_to_check = new_nodes_to_check

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            # assert self.nodes_dict.find(u) is not None

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            print("destination is not in Dijkstra graph")
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        # h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))
        # h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=None):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        open_heap = []
        heapq.heappush(open_heap, (0, (start[0], start[1])))

        while len(open_list) > 0:
            _, n = heapq.heappop(open_heap)
            n_coords = n
            node = self.nodes_dict.find(n).data

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()

                return path, np.round(length, 2)

            costs = np.linalg.norm(np.array(list(node.neighbor_set)).reshape(-1, 2) - [n_coords[0], n_coords[1]],
                                   axis=1)
            for cost, neighbor_node_coords in zip(costs, node.neighbor_set):
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8


class Node:
    def __init__(self, coords, frontiers, updating_map_info):
        self.coords = coords
        self.utility_range = parameter.UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        # possible neighbors are 5 * 5 surrounding nodes
        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1

        # neighbors including the node itself (for attention learning)
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)
            if self.utility <= parameter.MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict, hard_update=False):
        old_neighbor_set = deepcopy(self.neighbor_set)

        if hard_update:
            self.neighbor_matrix = -np.ones((5, 5))
            self.neighbor_matrix[2, 2] = 1
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(
                        np.array([self.coords[0] + (i - center_index) * parameter.NODE_RESOLUTION,
                                  self.coords[1] + (j - center_index) * parameter.NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    elif (neighbor_node.data.coords[0] <= updating_map_info.map_origin_x
                          or neighbor_node.data.coords[1] <= updating_map_info.map_origin_y
                          or neighbor_node.data.coords[0] >= updating_map_info.map_origin_x +
                          updating_map_info.map.shape[1] * updating_map_info.cell_size
                          or neighbor_node.data.coords[1] >= updating_map_info.map_origin_y +
                          updating_map_info.map.shape[0] * updating_map_info.cell_size):
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision_type(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if collision == parameter.FREE:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))
                        elif collision == parameter.OCCUPIED:
                            self.neighbor_matrix[i, j] = 0
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set:
                                self.neighbor_set.remove((neighbor_coords[0], neighbor_coords[1]))
                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 0
                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set:
                                neighbor_node.neighbor_set.remove((self.coords[0], self.coords[1]))
                        else:
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set:
                                self.neighbor_set.remove((neighbor_coords[0], neighbor_coords[1]))
                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set:
                                neighbor_node.neighbor_set.remove((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

        updated_neighbors = self.neighbor_set ^ old_neighbor_set
        updated_edges = set()
        coords = (self.coords[0], self.coords[1])
        for neighbor in updated_neighbors:
            updated_edges.add((coords, neighbor))

        return updated_edges

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info, global_frontiers):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if frontier not in global_frontiers:
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:
                self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility <= parameter.MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 0
        self.observable_frontiers = set()
        self.utility = 0
        self.need_update_neighbor = False


class KeyNode:
    def __init__(self, coords, utility, visited):
        self.coords = coords
        self.utility = utility
        self.visited = visited
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def add_neighbor_coords(self, neighbor_coords):
        self.neighbor_set.add(neighbor_coords)


class DStarLite:
    def __init__(self, nodes_dict, start, goal, max_t=0.1):
        self.nodes_dict = nodes_dict
        self.start = (start[0], start[1])
        self.goal = (goal[0], goal[1])
        self.km = 0
        self.rhs = dict()
        self.g = dict()
        self.U = []
        self.last = start
        self.initialize()
        self.first_run = True
        self.max_t = max_t
        self.start_t = None

    def initialize(self):
        self.km = 0
        self.rhs = dict()
        self.g = dict()
        self.U = []
        for node in self.nodes_dict.__iter__():
            node = node.data
            coords = (node.coords[0], node.coords[1])
            self.rhs[coords] = float('inf')
            self.g[coords] = float('inf')
        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def calculate_key(self, coords):
        g_rhs_min = min(self.g[coords], self.rhs[coords])
        return np.round(g_rhs_min + self.heuristic(self.start, coords) + self.km, 3), g_rhs_min

    def heuristic(self, coords1, coords2):
        x1, y1 = coords1[0], coords1[1]
        x2, y2 = coords2[0], coords2[1]
        return np.round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)
        # return np.round(abs(x2 - x1) + abs(y2 - y1), 3)

    def update_node(self, node):
        node_coords = (node.coords[0], node.coords[1])

        if node_coords != self.goal:
            min_rhs = float('inf')
            for neighbor_coords in node.neighbor_set:
                if node_coords != neighbor_coords:
                    cost = np.round(math.sqrt(
                        (node_coords[0] - neighbor_coords[0]) ** 2 + (node_coords[1] - neighbor_coords[1]) ** 2), 3)
                    min_rhs = min(min_rhs, np.round(self.g[neighbor_coords] + cost,3))
            self.rhs[node_coords] = min_rhs

        if node_coords in [item[1] for item in self.U]:
            self.U = [item for item in self.U if item[1] != node_coords]
            heapq.heapify(self.U)

        if self.g[node_coords] != self.rhs[node_coords]:
            heapq.heappush(self.U, (self.calculate_key(node_coords), node_coords))

    def compute_shortest_path(self):
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
            k_old, u = heapq.heappop(self.U)
            node = self.nodes_dict.find(u).data

            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
                continue

            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords == (node.coords[0], node.coords[1]):
                        pass
                    else:
                        neighbor = self.nodes_dict.find(neighbor_coords).data
                        self.update_node(neighbor)
            else:
                self.g[u] = float('inf')
                self.update_node(node)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords == (node.coords[0], node.coords[1]):
                        pass
                    else:
                        neighbor = self.nodes_dict.find(neighbor_coords).data
                        self.update_node(neighbor)
            if time.perf_counter() - self.start_t > self.max_t:
                return

    def fix_unconverged_path(self, unconverged_coords):
        while self.U and (self.U[0][0] < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start] or self.rhs[unconverged_coords] != self.g[unconverged_coords]):
            k_old, u = heapq.heappop(self.U)
            node = self.nodes_dict.find(u).data

            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
                continue

            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords == (node.coords[0], node.coords[1]):
                        pass
                    else:
                        neighbor = self.nodes_dict.find(neighbor_coords).data
                        self.update_node(neighbor)
            else:
                self.g[u] = float('inf')
                self.update_node(node)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords == (node.coords[0], node.coords[1]):
                        pass
                    else:
                        neighbor = self.nodes_dict.find(neighbor_coords).data
                        self.update_node(neighbor)
        if time.perf_counter() - self.start_t > self.max_t:
            return

    def get_shortest_path(self):
        self.start_t = time.perf_counter()
        self.compute_shortest_path()

        if time.perf_counter() - self.start_t > self.max_t:
            return [], 1e8

        path = []
        current = self.start
        current_node = self.nodes_dict.find(current).data
        dist = 0

        i = 0
        while current != self.goal:
            i += 1

            best_neighbor = None
            best_g = float('inf')
            for neighbor_coords in current_node.neighbor_set:
                if neighbor_coords != current:
                    g = self.g.get(neighbor_coords, float('inf'))
                    if g < best_g:
                        best_g = g
                        best_neighbor = neighbor_coords

            if len(path) >=2:
                if best_neighbor == path[-2]:
                    self.initialize()
                    self.compute_shortest_path()
                    path = []
                    current = self.start
                    current_node = self.nodes_dict.find(current).data
                    dist = 0
                    print("Path search failed! Recompute Dstar-lite")
                    continue

            if self.g[current] == float('inf') or best_neighbor is None:
                print("Path not found! Unreachable node encountered")
                return [], 1e8

            if time.perf_counter() - self.start_t > self.max_t:
                return [], 1e8

            path.append(best_neighbor)
            dist += np.round(math.sqrt((current[0] - best_neighbor[0]) ** 2 + (current[1] - best_neighbor[1]) ** 2), 3)
            current = best_neighbor
            current_node = self.nodes_dict.find(current).data

        return path, np.round(dist, 2)

    def update_graph(self, new_start=None, new_nodes=None, deleted_nodes=None, updated_edges=None):
        if new_start:
            self.start = new_start
            self.km += self.heuristic(self.last, self.goal) - self.heuristic(self.start, self.goal)
            self.last = self.start

        nodes_to_update = set()

        if deleted_nodes:
            deleted_coords = set()
            for node in deleted_nodes:
                node_coords = (node.coords[0], node.coords[1])

                del self.g[node_coords]
                del self.rhs[node_coords]
                deleted_coords.add(node_coords)

                neighbor_set = node.neighbor_set
                for neighbor_coords in neighbor_set:
                    if neighbor_coords != node_coords:
                        neighbor_node = self.nodes_dict.find(neighbor_coords)
                        if neighbor_node is not None:
                            neighbor_node = neighbor_node.data
                            nodes_to_update.add(neighbor_node)

            self.U = [item for item in self.U if item[1] not in deleted_coords]
            heapq.heapify(self.U)

        if new_nodes:
            for node in new_nodes:
                coords = (node.coords[0], node.coords[1])
                self.rhs[coords] = float('inf')
                self.g[coords] = float('inf')
                nodes_to_update.add(node)
                for neighbor_coords in node.neighbor_set:
                    if neighbor_coords != coords:
                        neighbor_node = self.nodes_dict.find(neighbor_coords).data
                        nodes_to_update.add(neighbor_node)

        for edge in updated_edges:
            u_coords, v_coords = edge[0], edge[1]
            u = self.nodes_dict.find(u_coords)
            if u:
                nodes_to_update.add(u.data)
            v = self.nodes_dict.find(v_coords)
            if v:
                nodes_to_update.add(v.data)

        if nodes_to_update:
            for node in nodes_to_update:
                self.update_node(node)


