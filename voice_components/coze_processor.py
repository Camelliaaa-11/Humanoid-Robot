#!/usr/bin/env python3
# coze_processor.py - 接收语音文本并调用Coze API，保存回复到文档

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import os
from datetime import datetime

class CozeProcessor(Node):
    def __init__(self):
        super().__init__('coze_processor')
        
        # 订阅语音识别文本
        self.subscription = self.create_subscription(
            String,
            '/voice_to_coze/text',
            self.text_callback,
            10
        )
        
        # 发布Coze回复（可选，用于其他节点）
        self.reply_publisher = self.create_publisher(
            String,
            '/coze_reply',
            10
        )
        
        # Coze API配置
        self.api_key = "pat_BFbY9RMkgmJOh2J36ZFKvEe4u8CzAc5uYCSwcLEk3g14NdqnLM1YEQn3CF4T1Pj0"
        self.bot_id = "7563240689564401691"
        self.user_id = "robot_user_001"
        
        # 创建日志目录
        self.log_dir = 'coze_conversations'
        self.reply_dir = 'coze_replies'  # 专门保存回复的目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.reply_dir, exist_ok=True)
        
        self.get_logger().info("Coze处理器已启动，等待语音输入...")

    def text_callback(self, msg):
        """处理接收到的语音文本"""
        user_text = msg.data.strip()
        
        if not user_text:
            self.get_logger().warn("收到空文本，跳过处理")
            return
            
        self.get_logger().info(f"收到语音文本: {user_text}")
        
        # 调用Coze API
        bot_reply = self.call_coze_api(user_text)
        
        if bot_reply:
            # 保存完整对话记录
            self.save_conversation(user_text, bot_reply)
            
            # 单独保存Coze回复到文档
            self.save_coze_reply(bot_reply)
            
            # 发布回复（可选）
            reply_msg = String()
            reply_msg.data = bot_reply
            self.reply_publisher.publish(reply_msg)
            
            self.get_logger().info(f"Coze回复已保存")

    def call_coze_api(self, message):
        """调用Coze API获取回复 - 使用能工作的流式方法"""
        url = "https://api.coze.cn/v3/chat"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "stream": True,
            "auto_save_history": True,
            "additional_messages": [
                {
                    "role": "user",
                    "content": message,
                    "content_type": "text"
                }
            ]
        }
        
        try:
            self.get_logger().info("正在调用Coze API...")
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            full_reply = ""
            self.get_logger().info("AI回复: ")
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if line_str.startswith('data:'):
                        data_str = line_str[5:].strip()
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if ('content' in data and 
                                isinstance(data['content'], str) and 
                                data['content'].strip() and
                                not data['content'].startswith('{')):
                                content = data['content']
                                print(content, end='', flush=True)
                                full_reply += content
                        except json.JSONDecodeError:
                            continue
            
            print()  # 换行
            
            # 在最终返回时去除重复
            cleaned_reply = self.remove_duplicates(full_reply)
            return cleaned_reply.strip()
            
        except requests.exceptions.Timeout:
            self.get_logger().error("API请求超时")
            return "抱歉，网络连接超时，请稍后重试。"
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"API调用错误: {e}")
            return "抱歉，服务暂时不可用，请稍后重试。"
        except Exception as e:
            self.get_logger().error(f"其他错误: {e}")
            return f"抱歉，处理请求时出现错误: {e}"

    def remove_duplicates(self, text):
        """去除文本中的重复内容"""
        if not text:
            return text
            
        # 如果文本较短，直接返回
        if len(text) < 20:
            return text
            
        # 检查是否整个文本重复了
        half_length = len(text) // 2
        first_half = text[:half_length]
        second_half = text[half_length:]
        
        # 如果后半部分与前半部分相同或几乎相同，只返回前半部分
        if first_half in second_half and len(first_half) > 10:
            self.get_logger().info("检测到重复内容，已去重")
            return first_half.strip()
        
        # 如果没有明显重复，返回原文本
        return text

    def save_conversation(self, user_text, bot_reply):
        """保存完整对话记录（用户问题 + AI回复）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"对话时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                f.write(f"用户问题: {user_text}\n")
                f.write("-" * 30 + "\n")
                f.write(f"AI回复: {bot_reply}\n")
                f.write("=" * 50 + "\n")
            
            self.get_logger().info(f"完整对话已保存: {filename}")
            
        except Exception as e:
            self.get_logger().error(f"保存对话失败: {e}")

    def save_coze_reply(self, bot_reply):
        """单独保存Coze回复到文档 - 只保存纯文本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # 保存为详细回复文件
        detail_filename = f"{self.reply_dir}/coze_reply_{timestamp}.txt"
    
        # 保存为最新回复文件（方便其他程序读取）
        latest_filename = f"{self.reply_dir}/latest_coze_reply.txt"
    
        try:
            # 保存详细回复文件 - 只保存纯文本，不加任何格式
            with open(detail_filename, 'w', encoding='utf-8') as f:
                f.write(bot_reply)  # 只写纯文本
        
            # 保存最新回复文件（只包含纯文本回复）
            with open(latest_filename, 'w', encoding='utf-8') as f:
                f.write(bot_reply)
        
            self.get_logger().info(f"Coze回复已保存: {detail_filename}")
            self.get_logger().info(f"最新回复已更新: {latest_filename}")
        
        except Exception as e:
            self.get_logger().error(f"保存Coze回复失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CozeProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
