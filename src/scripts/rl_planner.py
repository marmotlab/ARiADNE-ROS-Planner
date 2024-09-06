#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter("ignore", UserWarning)
import rospy
import rospkg
import numpy as np
import torch
import os
import time
from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from agent import Agent
from model import PolicyNet
from utils import *
import parameter


class Runner:
    def __init__(self):
        self.map_info = None
        self.device = 'cpu'
        self.step = 0

        # visualization
        self.publish_graph = rospy.get_param('~publish_graph', True)

        # map related
        parameter.CELL_SIZE = rospy.get_param('~map_resolution', parameter.CELL_SIZE)
        parameter.FREE = rospy.get_param('~map_free_value', parameter.FREE)
        parameter.OCCUPIED = rospy.get_param('~map_occupied_value', parameter.OCCUPIED)
        parameter.UNKNOWN = rospy.get_param('~map_unknown_value', parameter.UNKNOWN)

        # utility related
        parameter.SENSOR_RANGE = rospy.get_param('~sensor_range', parameter.SENSOR_RANGE)
        parameter.UTILITY_RANGE = rospy.get_param('~utility_range_factor', 0.5) * parameter.SENSOR_RANGE
        parameter.MIN_UTILITY = rospy.get_param('~min_utility', parameter.MIN_UTILITY)
        parameter.FRONTIER_CELL_SIZE = rospy.get_param('~frontier_downsample_factor', 1) * parameter.CELL_SIZE

        # graph related
        parameter.NODE_RESOLUTION = rospy.get_param('~node_resolution', parameter.NODE_RESOLUTION)
        parameter.CLUSTER_RANGE = rospy.get_param('~frontier_cluster_range', parameter.CLUSTER_RANGE)
        parameter.THR_NEXT_WAYPOINT = rospy.get_param('~next_waypoint_threshold', parameter.THR_NEXT_WAYPOINT)
        parameter.THR_GRAPH_HARD_UPDATE = rospy.get_param('~hard_update_threshold', parameter.THR_GRAPH_HARD_UPDATE)

        # replanning related
        parameter.THR_TO_WAYPOINT = rospy.get_param('~waypoint_threshold', parameter.THR_TO_WAYPOINT)
        parameter.AVOID_OSCILLATION = rospy.get_param('~avoid_waypoint_oscillation', parameter.AVOID_OSCILLATION)
        frequency = rospy.get_param('~replanning_frequency', 2.5)

        # network model file
        self.model_file = "checkpoint.pth"

        # robot coordination wrt map frame
        self.robot_location = None

        # the grid occupied by the robot
        self.robot_cell = None

        # initialize robot planner
        self.robot = None
        self.init_agent()

        # waypoint
        self.next_waypoint_list = []
        self.history_waypoint_list = []
        self.next_way_point = None

        # termination status
        self.done = False

        # subscribers
        rospy.Subscriber('/projected_map', OccupancyGrid, self.get_map_callback, queue_size=1)
        rospy.Subscriber('/state_estimation', Odometry, self.get_loc_callback, queue_size=1)

        # publishers
        self.way_point_pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
        self.run_time_pub = rospy.Publisher('/runtime', Float32, queue_size=1)
        self.edge_pub = rospy.Publisher('/edge', Marker, queue_size=1)
        self.node_pub = rospy.Publisher('/node', PointCloud2, queue_size=1)
        self.frontier_pub = rospy.Publisher('/frontier', PointCloud2, queue_size=1)
        
        # get map and robot location
        while self.map_info is None or self.robot_location is None:
            pass

        rate = rospy.Rate(20)
        rospy.Timer(rospy.Duration(1 / frequency), self.run)
        try:
            rate.sleep()
            rospy.spin()
        except KeyboardInterrupt:
            pass

    def get_map_callback(self, msg):
        t1 = time.time()
        delta = msg.info.resolution
        map_origin_x = msg.info.origin.position.x
        map_origin_y = msg.info.origin.position.y
        
        map_width = msg.info.width
        map_height = msg.info.height
        ros_map = np.array(np.array(msg.data).reshape(map_height, map_width).astype(np.int8))

        # padding the map with unknown area to avoid a frontier calculation issue
        pad_size = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        processed_map = np.pad(ros_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=parameter.UNKNOWN)
        map_origin_x -= delta * pad_size
        map_origin_y -= delta * pad_size
        robot_belief_map = processed_map

        self.map_info = MapInfo(robot_belief_map, map_origin_x, map_origin_y, delta)
        t2 = time.time()
        # print("process map using {}".format(t2 - t1))

    def get_loc_callback(self, msg):
        if self.map_info is None:
            return
        self.robot_location = np.around(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), 1)
        self.robot_cell = get_cell_position_from_coords(self.robot_location, self.map_info)

    def waypoint_wrapper(self, loc):
        way_point = PointStamped()
        way_point.header.frame_id = "map"
        way_point.header.stamp = rospy.Time.now()
        way_point.point.x = loc[0]
        way_point.point.y = loc[1]
        return way_point

    def init_agent(self):
        policy_net = PolicyNet(parameter.NODE_INPUT_DIM, parameter.EMBEDDING_DIM).to(self.device)
        model_folder = os.path.join(rospkg.RosPack().get_path('rl_planner'), 'scripts/model')
        model_file = os.path.join(model_folder, self.model_file)
        policy_net.load_state_dict(torch.load(model_file, map_location=self.device)['policy_model'])

        self.robot = Agent(policy_net, self.device, self.publish_graph)

    def run(self, event=None):
        # no more planning if exploration is completed
        if self.done:
             return

        # check and solve oscillation between two waypoints
        if parameter.AVOID_OSCILLATION and len(self.history_waypoint_list) > 4:
            if self.history_waypoint_list[-1] != self.history_waypoint_list[-2]:
                if self.history_waypoint_list[-1] == self.history_waypoint_list[-3] and self.history_waypoint_list[-2] == self.history_waypoint_list[-4]:
                    self.next_waypoint_list = []
                    rospy.logdebug("Waypoint oscillation detected")
                    if np.linalg.norm(self.next_way_point - self.robot_location) > parameter.THR_TO_WAYPOINT:
                        return
                    else:
                        rospy.logdebug("Waypoint oscillation solved")

        # if planned one more step, use it
        if len(self.next_waypoint_list) > 0:
            if np.linalg.norm(self.next_way_point - self.robot_location) > parameter.THR_TO_WAYPOINT:
                pass
            else:
                self.robot_location = self.next_way_point
                self.next_way_point = self.next_waypoint_list.pop(0)
                way_point_msg = self.waypoint_wrapper(self.next_way_point)
                self.way_point_pub.publish(way_point_msg)
        self.next_waypoint_list = []
        # print("robot location at", self.robot_location)

        # remove nodes on obstacles if any
        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)

        # find nearest node to the robot
        robot_node_location = self.robot_location
        if self.robot_location[0] != 0 or self.robot_location[1] != 0:
            nearest_node = self.robot.node_manager.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)[0]
            node_coords = nearest_node.data.coords
            robot_node_location = node_coords

        t1 = time.time()
        # updating planning graph
        self.robot.update_planning_state(self.map_info, robot_node_location)

        # check the termination status
        if sum(self.robot.key_utility) == 0:
            g = "\033[92m"
            n= "\033[0m"
            rospy.loginfo(f"{g}Exploration Completed{n}")
            self.done = True

        # get rl observation
        t2 = time.time()
        observation = self.robot.get_observation(self.robot_location)
        t3 = time.time()

        # network inference to get next waypoint
        next_location, next_node_index = self.robot.select_next_waypoint(observation)

        self.next_waypoint_list.append(next_location)
        if len(self.history_waypoint_list) > 0:
            if (next_location[0], next_location[1]) != self.history_waypoint_list[-1]:
                self.history_waypoint_list.append((next_location[0], next_location[1]))
        else:
            self.history_waypoint_list.append((next_location[0], next_location[1]))

        # planning one more step if next node's utility is zero
        if self.robot.node_manager.nodes_dict.find(next_location.tolist()).data.utility == 0:
            next_observation = self.robot.get_next_observation(next_node_index, observation)
            next_next_location, _ = self.robot.select_next_waypoint(next_observation)

            # if next waypoint is too close, go to the next next waypoint
            if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                self.next_waypoint_list = []

            self.next_waypoint_list.append(next_next_location)

        t4 = time.time()
        # print("next waypoint at", next_location)
        # print("update planning state using {}".format(t2 - t1))
        # print("prepare tensor input using {}".format(t3 - t2))
        # print("neural network inference using {}".format(t4-t3))

        # get waypoint message
        self.next_way_point = self.next_waypoint_list.pop(0)
        way_point_msg = self.waypoint_wrapper(self.next_way_point)

        # get planning time message
        run_time = Float32()
        if not self.done:
            run_time.data = t4 - t1
        else:
            run_time.data = 0

        # publish
        self.run_time_pub.publish(run_time)
        self.way_point_pub.publish(way_point_msg)

        self.step += 1
        if self.publish_graph:
            self.visualize_graph()

    def visualize_graph(self):
        # visualize edges
        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = rospy.Time.now()
        edges.type = Marker.LINE_LIST
        edges.scale.x = 0.1
        edges.color.r = 0.0
        edges.color.g = 0.6
        edges.color.b = 0.0
        edges.color.a = 1.0
        edges.pose.orientation.x = 0.0
        edges.pose.orientation.y = 0.0
        edges.pose.orientation.z = 0.0
        edges.pose.orientation.w = 1.0

        for coords in self.robot.key_node_coords:
            node = self.robot.node_manager.key_node_dict[(coords[0], coords[1])]
            for neighbor_coords in node.neighbor_set:
                start = Point()
                start.x = coords[0]
                start.y = coords[1]
                end_coords = (neighbor_coords - coords) / 2 + coords
                end = Point()
                end.x = end_coords[0]
                end.y = end_coords[1]
                edges.points.append(start)
                edges.points.append(end)

        self.edge_pub.publish(edges)

        # visualize nodes
        nodes = []
        for node_coords, utility in zip(self.robot.key_node_coords, self.robot.key_utility):
            nodes.append((node_coords[0], node_coords[1], 0.0, utility))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        nodes = point_cloud2.create_cloud(header, fields, nodes)
        self.node_pub.publish(nodes)

        # visualize frontiers
        frontiers = []
        for frontier in self.robot.frontier:
            frontiers.append((frontier[0], frontier[1], 0))
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        frontiers = point_cloud2.create_cloud(header, fields, frontiers)
        self.frontier_pub.publish(frontiers)
        

if __name__ == '__main__':
    rospy.init_node('rl_planner', anonymous=True)
    rl_runner = Runner()
