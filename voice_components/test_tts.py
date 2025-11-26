#!/usr/bin/env python3
"""
TTS模块测试脚本
用于单独测试TTS功能而不依赖ASR
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time

class TTSTester(Node):
    def __init__(self):
        super().__init__('tts_tester')
        
        # 发布测试文本到/coze_reply话题
        self.test_publisher = self.create_publisher(String, '/coze_reply', 10)
        
        self.get_logger().info("TTS测试节点启动，准备发送测试文本...")
        
    def send_test_text(self, text):
        """发送测试文本"""
        msg = String()
        msg.data = text
        self.test_publisher.publish(msg)
        self.get_logger().info(f"发送测试文本: {text}")

def main():
    rclpy.init()
    tester = TTSTester()
        
    # 等待节点初始化
    time.sleep(2)
    
    # 发送测试文本
    test_texts = [
        "你好，我是机器人",
        "开始执行动作",
        "检测到前方障碍物",
        "电量充足，可以继续工作"
    ]
    
    for text in test_texts:
        tester.send_test_text(text)
        time.sleep(5)  # 等待语音播放完成
    
    tester.get_logger().info("测试完成")
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()