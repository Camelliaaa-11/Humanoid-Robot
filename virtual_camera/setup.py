from setuptools import setup

package_name = 'virtual_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Virtual camera node for ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'virtual_camera_node = virtual_camera.virtual_camera_node:main',
        'image_capture_node = virtual_camera.image_capture_node:main',
        'rviz_image_capture_node = virtual_camera.rviz_image_capture_node:main',
        'coze_final_upload_node = virtual_camera.coze_final_upload_node:main',
    ],
},
)
