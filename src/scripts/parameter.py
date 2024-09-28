CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 2.0 # meter

FREE = 0
OCCUPIED = 100
UNKNOWN = -1

SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.5 * SENSOR_RANGE  # for each node, frontiers in this range will be considered as utility
MIN_UTILITY = 3  # if the number observable frontiers is less than this value, consider it is zero utility
FRONTIER_CELL_SIZE = 1 * CELL_SIZE  # downsample the frontiers based on this value

UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # the minimal map that contains all possible updating nodes

NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 25

THR_TO_WAYPOINT = 1 # meter, the waypoint will be considered as arrived if the robot is closer than this value
THR_NEXT_WAYPOINT = 5 # meter, the planner will try to plan a waypoint farther than this value
THR_GRAPH_HARD_UPDATE = 8 # meter, node and edges in this range will be fully updated

CLUSTER_RANGE = 10 # meter, frontiers will be clustered based on this range

AVOID_OSCILLATION = True # if the planner outputs back and forth waypoints, move to one of them
ENABLE_SAVE_MODE = False # if the planner outputs waypoints in loop, move to the nearest frontier
ENABLE_DSTARLITE = False # Use D*-lite for graph rarefaction instead of A*