#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class VirtualCameraNode(Node):
    def __init__(self):
        super().__init__('virtual_camera')
        
        # 创建图像发布者
        self.publisher_ = self.create_publisher(Image, '/virtual_camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.frame_count = 0
        
        self.get_logger().info('🎥 虚拟摄像头节点已启动，正在发布话题 /virtual_camera/image_raw')
    
    def generate_virtual_image(self, width=640, height=480):
        """生成虚拟测试图像"""
        # 创建渐变背景
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 简单的渐变效果
        for y in range(height):
            for x in range(width):
                image[y, x] = [
                    int(255 * x / width),           # 红色渐变
                    int(255 * y / height),          # 绿色渐变  
                    (x + y + self.frame_count) % 256  # 蓝色动态变化
                ]
        
        # 绘制移动的圆形
        center_x = 300 + int(100 * np.sin(self.frame_count * 0.1))
        center_y = 200 + int(100 * np.cos(self.frame_count * 0.1))
        cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), -1)
        
        # 添加文字信息
        cv2.putText(image, f'Virtual Camera Frame: {self.frame_count}', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, 'ROS2 Virtual Camera Demo', 
                   (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def timer_callback(self):
        """定时发布图像"""
        virtual_image = self.generate_virtual_image()
        
        try:
            # 转换为ROS消息并发布
            ros_image = self.bridge.cv2_to_imgmsg(virtual_image, encoding='rgb8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'virtual_camera_frame'
            
            self.publisher_.publish(ros_image)
            self.frame_count += 1
            
            if self.frame_count % 20 == 0:
                self.get_logger().info(f'📸 已发布 {self.frame_count} 帧图像')
                
        except Exception as e:
            self.get_logger().error(f'❌ 图像发布失败: {str(e)}')

def main():
    rclpy.init()
    node = VirtualCameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n🛑 虚拟摄像头节点已关闭')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
