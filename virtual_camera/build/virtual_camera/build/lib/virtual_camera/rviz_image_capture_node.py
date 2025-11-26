#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class RvizImageCaptureNode(Node):
    def __init__(self):
        super().__init__('rviz_image_capture_node')
        
        self.bridge = CvBridge()
        
        # 订阅虚拟摄像头话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        # 保存目录
        self.save_dir = "rviz_captured_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.get_logger().info('🎯 RViz2图像捕捉节点已启动')
        self.get_logger().info('这个节点保存的是RViz2中显示的相同图像数据')
        self.get_logger().info('按 Ctrl+C 停止，图像自动保存')
        
        # 自动保存计数器
        self.auto_save_count = 0
        self.max_saves = 10  # 最多保存10张图片
        
    def image_callback(self, msg):
        """自动保存接收到的图像"""
        if self.auto_save_count >= self.max_saves:
            return
            
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # 转换RGB到BGR用于保存
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            jpg_filename = f"{self.save_dir}/rviz_capture_{timestamp}.jpg"
            png_filename = f"{self.save_dir}/rviz_capture_{timestamp}.png"
            
            # 保存图像
            cv2.imwrite(jpg_filename, cv_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(png_filename, cv_image_bgr)
            
            self.auto_save_count += 1
            self.get_logger().info(f'💾 RViz2图像 #{self.auto_save_count} 已保存')
            self.get_logger().info(f'   文件: rviz_capture_{timestamp}.jpg/png')
            
            # 达到最大保存数量后自动关闭
            if self.auto_save_count >= self.max_saves:
                self.get_logger().info('✅ 已完成10张图像保存，节点将自动关闭')
                raise KeyboardInterrupt
                
        except Exception as e:
            self.get_logger().error(f'❌ 图像保存失败: {str(e)}')

def main():
    rclpy.init()
    node = RvizImageCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 RViz2图像捕捉节点已关闭')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
