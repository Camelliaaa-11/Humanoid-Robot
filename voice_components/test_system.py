#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import json

class SystemTester(Node):
    def __init__(self):
        super().__init__('system_tester')
        
        # 创建发布者
        self.text_publisher = self.create_publisher(
            String,
            '/voice_to_coze/text',
            10
        )
        
        # 创建订阅者来监听回复
        self.reply_subscriber = self.create_subscription(
            String,
            '/coze_reply',
            self.reply_callback,
            10
        )
        
        self.last_reply = None
        self.get_logger().info("系统测试器已启动")

    def send_test_message(self, text):
        """发送测试消息"""
        msg = String()
        msg.data = text
        self.text_publisher.publish(msg)
        self.get_logger().info(f"已发送测试消息: {text}")

    def reply_callback(self, msg):
        """接收Coze回复"""
        self.last_reply = msg.data
        self.get_logger().info(f"收到Coze回复: {msg.data}")

def main():
    rclpy.init()
    tester = SystemTester()
    
    # 等待节点建立连接
    print("等待系统连接建立...")
    time.sleep(3)
    
    # 测试消息列表
    test_messages = [
        "潮汕文化AI共创系统——以英歌文化为例的作者是谁"
    ]
    
    print("=== 系统测试开始 ===")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n测试 {i}/4: {message}")
        tester.send_test_message(message)
        
        # 等待回复
        tester.last_reply = None
        start_time = time.time()
        while time.time() - start_time < 15:  # 等待15秒
            rclpy.spin_once(tester, timeout_sec=0.1)
            if tester.last_reply:
                print(f"✅ 测试成功！回复: {tester.last_reply}")
                break
        else:
            print("❌ 测试超时，未收到回复")
        
        time.sleep(2)  # 等待2秒再进行下一个测试
    
    print("\n=== 系统测试完成 ===")
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
