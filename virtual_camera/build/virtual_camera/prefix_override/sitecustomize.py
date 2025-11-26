import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wwx/ros2_ws/src/virtual_camera/install/virtual_camera'
