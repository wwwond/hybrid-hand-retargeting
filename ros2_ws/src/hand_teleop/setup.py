from setuptools import setup

package_name = 'hand_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jaewon',
    maintainer_email='esuedp@gmail.com',
    description='비전 기반 로봇 손 텔레오퍼레이션',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hand_tracker = hand_teleop.hand_tracker_node:main',
            'robot_hand = hand_teleop.robot_hand_node:main',
        ],
    },
)
