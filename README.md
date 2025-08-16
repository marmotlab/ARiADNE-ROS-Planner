# ARiADNE-ROS-Planner

ARiADNE is **A** **R**e**i**nforcement learning apporach using **A**ttention-based **D**eep **N**etwork.
It is designed to tackle Lidar-based autonomous single-robot exploration in 2D action space.
ARiADNE builds an informative graph based on the partial exploration map as the input of the attention-based neural network.
The neural network trained by deep reinforcement learning will select one of the neighboring nodes as the next waypoint iteratively. 
The robot will follow the planned waypoint to explore the environment.
The planner is trained to estimate the long-term exploration efficiency of each potential waypoint and find a policy that can maximize the expectation of the predicted exploration efficiency.
The informative graph can be rarefied to make it applicable in relatively larger-scale applications.

If you are interested in more details, please check our related publications in [ICRA2023](https://arxiv.org/pdf/2301.11575) and [RAL](https://arxiv.org/pdf/2403.10833).

This repo contains the source code for ARiADNE planner in ROS1 and [ROS2](https://github.com/marmotlab/ARiADNE-ROS-Planner/tree/humble)

<p align="center">
<img src="demo/example.jpg" width="480"/>
</p>

**16 Aug 2025**: support ROS2 Humble.

**5 Jan 2025**: support random starting location.

## Demo
Here is a demo video showing ARiADNE planner exploring the indoor environment provided by [TARE](https://github.com/caochao39/tare_planner/tree/melodic-noetic). 
The video is playing at the original speed.
This experiment was conducted on an ASUS mini PC with an Intel i7-12700H CPU. 

https://github.com/user-attachments/assets/e4aecdb2-9c6e-4803-996b-25efc4cae221

In this demo, the robot traveled 948m in 545s when the exploration was completed. 
The average planning time was 0.17s.
The average waypoint publishing frequency was around 2.5Hz. 
You can find the per-step record [here](demo/metrics.txt). 
The final exploration trajectory is shown below.    

<p align="center">
<img src="demo/demo_indoor.png" width="640"/>
</p>

The largest environment we tested is the tunnel environment as below:

https://github.com/user-attachments/assets/6d4465eb-38fb-4fb5-943d-4e9a9953c75a

## Usage
### 1. Prerequisites
We tested this planner on Ubuntu 18.04 ROS [Melodic](https://wiki.ros.org/melodic/Installation) and Ubuntu 20.04 ROS [Noetic](http://wiki.ros.org/noetic/Installation).
In particular, our planner relies on [Octomap](https://octomap.github.io/) to transfer pointcloud to occupancy grid map:
```
sudo apt-get install ros-noetic-octomap
```
We recommend to use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#) for package management. 
Our planner is coded in Python and based on [Pytorch](https://pytorch.org/get-started/locally/).
Other than Pytorch, please install following packages by:
```
pip install scikit-image matplotlib
```
We tested our planner in various version of these packages so you can just install the latest one.
Then you can download this repo and compile it.
```
git clone https://github.com/marmotlab/ARiADNE-ROS-Planner.git
cd ARiADNE-ROS-Planner
catkin_make
```
**Note:** We only use CPU to do the network inference, so you do not need a GPU.

### 2. Development environments
In practice, our planner needs to cooperate with a Lidar SLAM module which outputs sensor odometry and Lidar scan, and a waypoint follower module which navigates the robot to the planned waypoint.
Fortunately, you can test our planner easily in the development environments provided by [CMU Robotics Institute](https://www.cmu-exploration.com/development-environment).

Please follow instructions for [CMU Development Environment](https://www.cmu-exploration.com/development-environment) to set up the Gazebo simulation, SLAM module, and waypoint follower module.

### 3. Run the code
To run the development environment, go to the development environment folder in a terminal and run:
```
source devel/setup.bash 
roslaunch vehicle_simulator system_indoor.launch
```
Our planner can work in three of their environments: indoor, forest, and tunnel.

To run ARiADNE planner, go to the planner folder in another terminal (launch your conda environment if any) and run:
```
source devel/setup.bash 
roslaunch rl_planner rl_planner.launch
```
if running the planner in the forest environment, run:
```
roslaunch rl_planner rl_planner_forest.launch
```
if running the planner in the tunnel environment, run:
```
roslaunch rl_planner rl_planner_tunnel.launch
```
### 4. Test in other environments
~~As long as the starting coordination of the robot is $(0, 0)$,~~ you should be able to run this code directly.
However, to get better performance in different environments, you most likely need to tune some parameters in the launch file, such as the node resolution, the frontier downsample factor, and maybe the replanning frequency.
Some brief introduction of these parameters can be found in ``parameter.py.``
Here are examples of applying ARiADNE planner in CMU forest and tunnel and two indoor scenarios provided by [FAEL](https://github.com/SYSU-RoboticsLab/FAEL/tree/main).

<p align="center">
<img src="demo/forest.jpg" width="240" align="left"><img src="demo/fael_scenario3.jpg" width="240"><img src="demo/fael_scenario4.jpg" width="240" align="right">
</p>
<p align="center">
<img src="demo/tunnel.jpg" width="560" />
</p>

### 5. Train your own networks
You can train your own networks using [ARiADNE](https://github.com/marmotlab/ARiADNE) or its [ground truth critic variant](https://github.com/marmotlab/large-scale-DRL-exploration). To run the trained model, replace the checkpoint file under the model folder.

## Citation
If you find our work helpful or enlightening, feel free to cite our paper:

```
@inproceedings{cao2023ariadne,
  title={Ariadne: A reinforcement learning approach using attention-based deep networks for exploration},
  author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10219--10225},
  year={2023},
  organization={IEEE}
}
```

```
@article{cao2024deep,
  title={Deep Reinforcement Learning-based Large-scale Robot Exploration},
  author={Cao, Yuhong and Zhao, Rui and Wang, Yizhuo and Xiang, Bairan and Sartoretti, Guillaume},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```

## Author
[Yuhong Cao](https://www.yuhongcao.online)

**Note:** I am seeking to implement this planner in C++. 
Feel free to contact me if you are interested in collaboration.

## Credit
[Development environment](https://www.cmu-exploration.com/development-environment) is from CMU.

[Octomap](https://octomap.github.io/) is from University of Freiburg.

[Quad tree](https://github.com/toastdriven/quads) is from [Daniel Lindsley](https://github.com/toastdriven).

[ChatGPT](https://chatgpt.com/) also contributes some code here and there.

