import time
import rospy
import heapq
from copy import deepcopy
import parameter
import numpy as np
from utils import *
# from parameter import *
import quads


class NodeManager:
    def __init__(self):
        # use quad tree to index nodes in the original graph
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        # use dictionary to index nodes in the rarefied graph
        self.key_node_dict = {}

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
                self.remove_node_from_dict(node)
            elif len(node.neighbor_set) <= 1:
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
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    parameter.SENSOR_RANGE + parameter.NODE_RESOLUTION * 3):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)
            elif np.linalg.norm(node.coords - robot_location) < parameter.THR_GRAPH_HARD_UPDATE:
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict, hard_update=True)
        t3 = time.time()
        # print("update edges", t3 - t2)

        # remove nodes unconnected to the current node
        self.remove_unconnected_nodes(robot_location)

    def remove_unconnected_nodes(self, robot_location):
        all_nodes_coords = set()
        for node in self.nodes_dict.__iter__():
            node = node.data
            all_nodes_coords.add((node.coords[0], node.coords[1]))

        current_node = self.nodes_dict.find(robot_location.tolist()).data
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

        if len(all_connected_nodes) <= 1:
            # somehow the current node is the one should be removed
            rospy.logwarn("Not enough connected nodes! skip removing unconnected nodes")
            return

        for node in not_connected_nodes:
            node = self.nodes_dict.find(node).data
            self.remove_node_from_dict(node)

    def get_rarefied_graph(self, robot_location, map_info):
        t1 = time.time()
        # get all nodes with non-zero utility
        non_zero_utility_nodes = set()
        for node in self.nodes_dict.__iter__():
            if node.data.utility > 0:
                non_zero_utility_nodes.add(node.data)

        # we are going to cluster them
        cluster_center_nodes = set()
        clustered_non_zero_utility_nodes = set()

        # first find the current node and add it to the rarefied graph
        self.key_node_dict = {}
        current_node = self.nodes_dict.find(robot_location.tolist()).data
        key_current_node = KeyNode(current_node.coords, current_node.utility, current_node.visited)
        self.key_node_dict[(key_current_node.coords[0], key_current_node.coords[1])] = key_current_node

        # The first cluster center is the current node
        cluster_boundary = get_quad_tree_box(current_node.coords, 1 * parameter.SENSOR_RANGE)
        nodes_in_cluster = self.nodes_dict.within_bb(cluster_boundary)

        # we find paths to all non-zero-utility nodes in the first cluster and add them to the rarefied graph
        for node in nodes_in_cluster:
            if node.data.utility > 0:

                if (node.data.coords[0], node.data.coords[1]) in self.key_node_dict.keys():
                    non_zero_utility_nodes.remove(node.data)
                    continue

                path, dist = self.a_star(current_node.coords, node.data.coords)
                if dist == 1e8:
                    continue

                non_zero_utility_nodes.remove(node.data)

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
                            key_node = self.key_node_dict[(key_node_coords[0], key_node_coords[1])]
                        else:
                            node = self.nodes_dict.find(key_node_coords).data
                            key_node = KeyNode(node.coords, node.utility, node.visited)
                            self.key_node_dict[(key_node_coords[0], key_node_coords[1])] = key_node

        # find other cluster centers through cluster range and nodes' connectivity
        for center_node in non_zero_utility_nodes:
            if center_node in clustered_non_zero_utility_nodes:
                continue

            cluster_center_nodes.add(center_node)
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

        t2 = time.time()
        # print("find cluster center", t2 - t1)

        t = 0
        # for other cluster we only calculate the path to the cluster center
        for center_node in cluster_center_nodes:
            center_node_coords = self.nodes_dict.find(center_node.coords.tolist()).data.coords
            t11 = time.time()
            path, dist = self.a_star(robot_location, center_node_coords)
            t12 = time.time()

            # print(t12 - t11)
            t += t12 - t11
            if dist == 1e8:
                continue

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

        t3 = time.time()
        # assert len(self.key_node_dict) > 1
        # print("get sparse graph", t3 - t2, t)

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
                            self.neighbor_matrix[i, j] = 1
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set:
                                self.neighbor_set.remove((neighbor_coords[0], neighbor_coords[1]))
                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set:
                                neighbor_node.neighbor_set.remove((self.coords[0], self.coords[1]))
                        else:
                            if (neighbor_coords[0], neighbor_coords[1]) in self.neighbor_set:
                                self.neighbor_set.remove((neighbor_coords[0], neighbor_coords[1]))
                            if (self.coords[0], self.coords[1]) in neighbor_node.neighbor_set:
                                neighbor_node.neighbor_set.remove((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

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
