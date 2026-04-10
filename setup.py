from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tercom_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.py')) +
            glob(os.path.join('launch', '*.xml')) +
            glob(os.path.join('launch', '*.yaml'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.rviz'))),
        (os.path.join('share', package_name, 'scripts'),
            glob(os.path.join('scripts', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='TERCOM-based GPS-denied navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dem_server = tercom_nav.nodes.dem_server_node:main',
            'tercom_node = tercom_nav.nodes.tercom_node:main',
            'eskf_node = tercom_nav.nodes.eskf_node:main',
            'diagnostics_node = tercom_nav.nodes.diagnostics_node:main',
        ],
    },
)
