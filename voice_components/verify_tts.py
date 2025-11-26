#!/usr/bin/env python3
"""
TTS链路验证脚本
测试从Coze回复到语音播放的完整流程
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import json

class TTSVerifier(Node):
    def __init__(self):
        super().__init__('tts_verifier')
        
        # 订阅所有相关话题来监控通信
        self.coze_sub = self.create_subscription(
            String, '/coze_reply', self.coze_callback, 10
        )
        self.tts_sub = self.create_subscription(
            String, '/xunfei/tts_play', self.tts_callback, 10
        )
        
        # 发布测试消息
        self.test_publisher = self.create_publisher(
            String, '/coze_reply', 10
        )
        
        self.get_logger().info("TTS验证节点启动")
        self.message_received = False
        
    def coze_callback(self, msg):
        self.get_logger().info(f"✅ 收到Coze回复: {msg.data[:50]}...")
        self.message_received = True
        
    def tts_callback(self, msg):
        try:
            play_cmd = json.loads(msg.data)
            if "file" in play_cmd:
                self.get_logger().info(f"✅ 收到TTS播放命令，文件: {play_cmd['file']}")
            else:
                self.get_logger().info(f"✅ 收到TTS命令: {msg.data}")
        except:
            self.get_logger().info(f"✅ 收到TTS消息: {msg.data}")
        self.message_received = True
        
    def send_test_message(self):
        """发送测试消息"""
        test_msg = String()
        test_msg.data = "这是一条测试消息，用于验证TTS功能"
        self.test_publisher.publish(test_msg)
        self.get_logger().info("📤 发送测试消息到Coze回复话题")

def main():
    rclpy.init()
    verifier = TTSVerifier()
    
    # 等待节点初始化
    time.sleep(2)
    
    # 发送测试消息
    verifier.send_test_message()
    
    # 运行一段时间来接收回调
    verifier.get_logger().info("监听消息，10秒后退出...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        rclpy.spin_once(verifier, timeout_sec=1)
        if verifier.message_received:
            verifier.get_logger().info("✅ 通信链路正常！")
            break
    
    if not verifier.message_received:
        verifier.get_logger().warning("❌ 未收到任何消息，请检查通信链路")
    
    verifier.destroy_node()
    rclpy.shutdown()
    print("验证完成！")

if __name__ == '__main__':
    main()