#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter("ignore", UserWarning)

import rclpy
from rclpy.node import Node
import numpy as np
from numpy import pad
import torch
import os
import time
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from .agent import Agent
from .model import PolicyNet
from .node_manager import NodeManager
from .utils import *
from . import parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy


class Runner(Node):
    def __init__(self):
        super().__init__('rl_planner')
        self.map_info = None
        self.device = 'cpu'
        self.step = 0

        self.declare_parameter('publish_graph', True)
        self.publish_graph = self.get_parameter('publish_graph').value
        
        self.declare_parameter('map_resolution', parameter.CELL_SIZE)
        parameter.CELL_SIZE = self.get_parameter('map_resolution').value
        
        self.declare_parameter('map_free_value', parameter.FREE)
        parameter.FREE = self.get_parameter('map_free_value').value
        
        self.declare_parameter('map_occupied_value', parameter.OCCUPIED)
        parameter.OCCUPIED = self.get_parameter('map_occupied_value').value
        
        self.declare_parameter('map_unknown_value', parameter.UNKNOWN)
        parameter.UNKNOWN = self.get_parameter('map_unknown_value').value
        
        self.declare_parameter('sensor_range', parameter.SENSOR_RANGE)
        parameter.SENSOR_RANGE = self.get_parameter('sensor_range').value
        
        self.declare_parameter('utility_range_factor', 0.5)
        utility_range_factor = self.get_parameter('utility_range_factor').value
        parameter.UTILITY_RANGE = utility_range_factor * parameter.SENSOR_RANGE
        
        self.declare_parameter('min_utility', parameter.MIN_UTILITY)
        parameter.MIN_UTILITY = self.get_parameter('min_utility').value
        
        self.declare_parameter('frontier_downsample_factor', 1)
        frontier_downsample_factor = self.get_parameter('frontier_downsample_factor').value
        parameter.FRONTIER_CELL_SIZE = frontier_downsample_factor * parameter.CELL_SIZE
        
        self.declare_parameter('node_resolution', parameter.NODE_RESOLUTION)
        parameter.NODE_RESOLUTION = self.get_parameter('node_resolution').value
        
        self.declare_parameter('frontier_cluster_range', parameter.CLUSTER_RANGE)
        parameter.CLUSTER_RANGE = self.get_parameter('frontier_cluster_range').value
        
        self.declare_parameter('next_waypoint_threshold', parameter.THR_NEXT_WAYPOINT)
        parameter.THR_NEXT_WAYPOINT = self.get_parameter('next_waypoint_threshold').value
        
        self.declare_parameter('hard_update_threshold', parameter.THR_GRAPH_HARD_UPDATE)
        parameter.THR_GRAPH_HARD_UPDATE = self.get_parameter('hard_update_threshold').value
        
        self.declare_parameter('waypoint_threshold', parameter.THR_TO_WAYPOINT)
        parameter.THR_TO_WAYPOINT = self.get_parameter('waypoint_threshold').value
        
        self.declare_parameter('avoid_waypoint_oscillation', parameter.AVOID_OSCILLATION)
        parameter.AVOID_OSCILLATION = self.get_parameter('avoid_waypoint_oscillation').value
        
        self.declare_parameter('enable_save_mode', parameter.ENABLE_SAVE_MODE)
        parameter.ENABLE_SAVE_MODE = self.get_parameter('enable_save_mode').value
        
        self.declare_parameter('enable_dstarlite', parameter.ENABLE_DSTARLITE)
        parameter.ENABLE_DSTARLITE = self.get_parameter('enable_dstarlite').value
        
        self.declare_parameter('replanning_frequency', 2.5)
        frequency = self.get_parameter('replanning_frequency').value

        self.model_file = "checkpoint.pth"

        self.robot_location = None
        self.robot_cell = None
        self.robot = None
        self.start = None
        self.next_waypoint_list = []
        self.history_waypoint_list = []
        self.next_waypoint = None
        self.done = False
        self.save_mode = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/projected_map',
            self.get_map_callback,
            qos_profile=qos
        )
        self.loc_sub = self.create_subscription(
            Odometry,
            '/state_estimation',
            self.get_loc_callback,
            qos_profile=qos
        )
        
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.run_time_pub = self.create_publisher(Float32, '/runtime', 10)
        self.edge_pub = self.create_publisher(Marker, '/edge', 10)
        self.node_pub = self.create_publisher(PointCloud2, '/node', 10)
        self.frontier_pub = self.create_publisher(PointCloud2, '/frontier', 10)
        
        self.init_agent()
        
        self.timer = self.create_timer(1.0 / frequency, self.run)
        
        self.get_logger().info("Waiting for map and location data...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info("RL Planner initialized")

    def run(self):
        if self.map_info is None or self.robot_location is None:
            return 
        t1 = time.time()
        if self.done:
            return

        if self.save_mode:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                return
            else:
                if len(self.next_waypoint_list) > 0:
                    next_waypoint = self.next_waypoint_list.pop(0)
                    while (not check_collision(self.robot_location, np.array(next_waypoint), self.map_info) and 
                           np.linalg.norm(self.robot_location - np.array(next_waypoint)) < (parameter.THR_NEXT_WAYPOINT + parameter.NODE_RESOLUTION) and 
                           len(self.next_waypoint_list) > 0):
                        next_waypoint = self.next_waypoint_list.pop(0)
                    
                    self.next_waypoint = next_waypoint
                    self.history_waypoint_list.append((self.next_waypoint[0], self.next_waypoint[1]))
                    
                    waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
                    self.waypoint_pub.publish(waypoint_msg)
                    
                    run_time = Float32()
                    run_time.data = time.time() - t1
                    self.run_time_pub.publish(run_time)
                    return
                else:
                    self.save_mode = False
                    self.get_logger().warning("Switch back to RL")

        if parameter.AVOID_OSCILLATION and len(self.history_waypoint_list) > 4:
            if (self.history_waypoint_list[-1] == self.history_waypoint_list[-3] and 
                self.history_waypoint_list[-2] == self.history_waypoint_list[-4]):
                self.next_waypoint_list = []
                if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                    return

        if len(self.next_waypoint_list) > 0:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                pass
            else:
                self.robot_location = self.next_waypoint
                self.next_waypoint = self.next_waypoint_list.pop(0)
                waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
                self.waypoint_pub.publish(waypoint_msg)
        
        self.next_waypoint_list = []
        self.get_logger().debug(f"Robot location at {self.robot_location}")

        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)

        robot_node_location = self.robot_location
        if self.robot_location[0] != self.start[0] or self.robot_location[1] != self.start[1]:
            if len(self.robot.node_manager.nodes_dict) == 0:
                robot_node_location = self.start
            else:
                nearest_node = self.robot.node_manager.nodes_dict.nearest_neighbors(
                    self.robot_location.tolist(), 1)[0]
                node_coords = nearest_node.data.coords
                robot_node_location = node_coords

        self.robot.update_planning_state(self.map_info, robot_node_location)

        if sum(self.robot.key_utility) == 0:
            self.get_logger().info("\033[92mExploration Completed\033[0m")
            self.done = True
            run_time = Float32()
            run_time.data = 0.0
            self.run_time_pub.publish(run_time)
            return

        t2 = time.time()
        observation = self.robot.get_observation(self.robot_location)
        t3 = time.time()

        next_location, next_node_index = self.robot.select_next_waypoint(observation)

        self.next_waypoint_list.append(next_location)
        if len(self.history_waypoint_list) > 0:
            if (next_location[0], next_location[1]) != self.history_waypoint_list[-1]:
                self.history_waypoint_list.append((next_location[0], next_location[1]))
        else:
            self.history_waypoint_list.append((next_location[0], next_location[1]))

        if self.robot.node_manager.nodes_dict.find(next_location.tolist()).data.utility == 0:
            next_observation = self.robot.get_next_observation(next_node_index, observation)
            next_next_location, _ = self.robot.select_next_waypoint(next_observation)

            if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                self.next_waypoint_list = []

            self.next_waypoint_list.append(next_next_location)

        t4 = time.time()
        self.get_logger().debug(f"Next waypoint at {next_location}")
        self.get_logger().debug(f"Update planning state using {t2-t1:.4f}s")
        self.get_logger().debug(f"Prepare tensor input using {t3-t2:.4f}s")
        self.get_logger().debug(f"Neural network inference using {t4-t3:.4f}s")

        if parameter.ENABLE_SAVE_MODE:
            if self.detect_waypoint_loop():
                self.next_waypoint_list = self.robot.node_manager.path_to_nearest_frontier
                self.save_mode = True
                self.get_logger().warning("Switch to save mode")

        self.next_waypoint = self.next_waypoint_list.pop(0)
        waypoint_msg = self.waypoint_wrapper(self.next_waypoint)

        run_time = Float32()
        run_time.data = t4 - t1

        self.run_time_pub.publish(run_time)
        self.waypoint_pub.publish(waypoint_msg)

        self.step += 1
        if self.publish_graph:
            self.visualize_graph()

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
        processed_map = pad(ros_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=parameter.UNKNOWN)
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
        if self.start is None:

            x = np.array([(self.robot_location[0] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION, (self.robot_location[0] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION])
            y = np.array([(self.robot_location[1] // parameter.NODE_RESOLUTION) * parameter.NODE_RESOLUTION, (self.robot_location[1] // parameter.NODE_RESOLUTION + 1) * parameter.NODE_RESOLUTION])
            t1, t2 = np.meshgrid(x, y)
            candidate_starts = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
            dis_robot = np.linalg.norm(candidate_starts - self.robot_location, axis=1)
            sorted_candidate_starts = candidate_starts[np.argsort(dis_robot)]

            for start in sorted_candidate_starts:
                if is_free(start, self.map_info):
                    self.start = start
                    break

            self.start = np.around(self.start, 1)
            self.robot.node_manager = NodeManager(self.start)
            print("initialize quad tree at", self.start)
            print("initialize robot location at", self.robot_location)
        self.robot_cell = get_cell_position_from_coords(self.robot_location, self.map_info)

    def waypoint_wrapper(self, loc):
        way_point = PointStamped()
        way_point.header.frame_id = "map"
        way_point.header.stamp = self.get_clock().now().to_msg()
        way_point.point.x = loc[0]
        way_point.point.y = loc[1]
        return way_point

    def init_agent(self):
        policy_net = PolicyNet(parameter.NODE_INPUT_DIM, parameter.EMBEDDING_DIM).to(self.device)
        package_share_dir = get_package_share_directory('rl_planner')
        model_file = os.path.join(package_share_dir, 'model', self.model_file)
        print("launch model from:", model_file)
        
        if not os.path.exists(model_file):
            self.get_logger().error(f"Model file not found: {model_file}")
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        policy_net.load_state_dict(torch.load(model_file, map_location=self.device)['policy_model'])
        self.robot = Agent(policy_net, self.device, self.publish_graph)

    def detect_waypoint_loop(self, max_length=6):
        if len(self.history_waypoint_list) < max_length:
            return False

        waypoint_list_to_check = self.history_waypoint_list[-max_length:]
        loop =[]
        for i, waypoint in enumerate(waypoint_list_to_check[:-1]):
            if waypoint == waypoint_list_to_check[-1]:
                loop = waypoint_list_to_check[i:]

        if loop:
            loop_length = len(loop)
            if len(self.history_waypoint_list) < 2 * loop_length + 1:
                return False
            waypoint_list_to_check2 = self.history_waypoint_list[-max_length-loop_length+1:-loop_length+1]
            # print("length check", waypoint_list_to_check2, loop)
            loop2 = []
            for i, waypoint in enumerate(waypoint_list_to_check2[:-1]):
                if waypoint == waypoint_list_to_check2[-1]:
                    loop2 = waypoint_list_to_check2[i:]
                    break
            if loop2:
                return True
            else:
                return False

    def visualize_graph(self):
        # visualize edges
        edges = Marker()
        edges.header.frame_id = 'map'
        edges.header.stamp = self.get_clock().now().to_msg()
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
        header.stamp = self.get_clock().now().to_msg()
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
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        frontiers = point_cloud2.create_cloud(header, fields, frontiers)
        self.frontier_pub.publish(frontiers)

def main(args=None):
    rclpy.init(args=args)
    rl_runner = None
    try:
        rl_runner = Runner()
        rclpy.spin(rl_runner)
    except KeyboardInterrupt:
        print("Shutting down RL Planner...")
    except Exception as e:
        print(f"Exception in RL Planner: {e}")
    finally:
        if rl_runner is not None:
            rl_runner.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
