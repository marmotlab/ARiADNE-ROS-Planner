from setuptools import setup

package_name = 'rl_planner'
import os 
from glob import glob  

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'model'), glob('rl_planner/model/*')),
    ],
    install_requires=['setuptools','sensor_msgs_py'],
    zip_safe=True,
    maintainer='Yuhong Cao',
    description='Ariadene ROS2 planner',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "rl_planner = rl_planner.rl_planner:main",
        ],
    },
)
