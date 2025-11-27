#!/usr/bin/env python3
# coze_processor_textandimage.py - 修改版：通过Ubuntu中转图文数据

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import os
from datetime import datetime
import glob
import base64
import threading

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
        
        # 发布Coze回复
        self.reply_publisher = self.create_publisher(
            String,
            '/coze_reply',
            10
        )
        
        # Coze API配置
        self.api_key = "pat_BFbY9RMkgmJOh2J36ZFKvEe4u8CzAc5uYCSwcLEk3g14NdqnLM1YEQn3CF4T1Pj0"
        self.bot_id = "7563240689564401691"
        self.user_id = "robot_user_001"
        
        # 图片目录配置
        self.home_dir = os.path.expanduser("~")
        self.image_dir = os.path.join(self.home_dir, "voice_project", "image_robot")
        
        # 修改：在Ubuntu上启动本地HTTP服务器来提供图片
        self.local_server_port = 8000
        self.local_server_url = f"http://localhost:{self.local_server_port}"
        
        # 创建日志目录
        self.log_dir = os.path.join(self.home_dir, "voice_project", "coze_conversations")
        self.reply_dir = os.path.join(self.home_dir, "voice_project", "coze_replies")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.reply_dir, exist_ok=True)
        
        # 启动本地HTTP服务器（在后台线程中）
        self.start_local_server()
        
        # 启动ngrok隧道（可选，如果需要从外部访问）
        self.ngrok_url = None
        self.setup_ngrok_tunnel()
        
        self.get_logger().info("Coze图文问答处理器已启动，等待语音输入...")
        self.get_logger().info(f"图片目录: {self.image_dir}")
        self.get_logger().info(f"本地服务器: {self.local_server_url}")
        if self.ngrok_url:
            self.get_logger().info(f"Ngrok隧道: {self.ngrok_url}")

    def start_local_server(self):
        """启动一个简单的HTTP服务器来提供图片"""
        try:
            import http.server
            import socketserver
            
            # 切换到图片目录
            os.chdir(self.image_dir)
            
            # 在后台线程中启动HTTP服务器
            def run_server():
                handler = http.server.SimpleHTTPRequestHandler
                with socketserver.TCPServer(("", self.local_server_port), handler) as httpd:
                    self.get_logger().info(f"本地图片服务器启动在端口 {self.local_server_port}")
                    httpd.serve_forever()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
        except Exception as e:
            self.get_logger().warning(f"启动本地HTTP服务器失败: {e}")

    def setup_ngrok_tunnel(self):
        """设置ngrok隧道（可选）"""
        try:
            # 检查ngrok是否在运行
            result = os.popen("pgrep ngrok").read()
            if not result.strip():
                self.get_logger().info("启动ngrok隧道...")
                # 在后台启动ngrok
                os.system(f"ngrok http {self.local_server_port} > /dev/null 2>&1 &")
                # 等待ngrok启动
                import time
                time.sleep(3)
            
            # 获取ngrok公共URL（简化版本，实际可能需要调用ngrok API）
            self.ngrok_url = "https://your-tunnel.ngrok-free.app"  # 需要实际获取
            self.get_logger().info("Ngrok隧道已准备就绪")
            
        except Exception as e:
            self.get_logger().warning(f"设置ngrok隧道失败: {e}")

    def text_callback(self, msg):
        """处理接收到的语音文本"""
        user_text = msg.data.strip()
        
        if not user_text:
            self.get_logger().warn("收到空文本，跳过处理")
            return
            
        self.get_logger().info(f"收到语音文本: {user_text}")
        
        # 方案1：使用base64直接编码图片
        image_data = self.get_latest_image_base64()
        if not image_data:
            self.get_logger().error("未找到可用的图片")
            return
            
        self.get_logger().info("成功获取图片数据，准备调用Coze API...")
        
        # 调用Coze API进行图文问答
        bot_reply = self.call_coze_api_with_image(user_text, image_data)
        
        if bot_reply:
            # 保存完整对话记录
            self.save_conversation(user_text, bot_reply)
            
            # 单独保存Coze回复到文档
            self.save_coze_reply(bot_reply)
            
            # 发布回复
            reply_msg = String()
            reply_msg.data = bot_reply
            self.reply_publisher.publish(reply_msg)
            
            self.get_logger().info(f"Coze回复已发布和保存")

    def get_latest_image_base64(self):
        """获取最新图片的base64编码"""
        try:
            # 检查图片目录是否存在
            if not os.path.exists(self.image_dir):
                self.get_logger().error(f"图片目录不存在: {self.image_dir}")
                return None
            
            # 查找图片文件
            jpg_files = glob.glob(os.path.join(self.image_dir, "*.jpg"))
            png_files = glob.glob(os.path.join(self.image_dir, "*.png"))
            all_files = jpg_files + png_files
            
            if not all_files:
                self.get_logger().error(f"在目录 {self.image_dir} 中未找到图片文件")
                return None
            
            # 获取最新文件
            latest_file = max(all_files, key=os.path.getmtime)
            self.get_logger().info(f"使用图片: {os.path.basename(latest_file)}")
            
            # 读取图片并转换为base64
            with open(latest_file, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            return image_data
            
        except Exception as e:
            self.get_logger().error(f"获取图片数据失败: {e}")
            return None

    def get_latest_image_url(self):
        """备选方案：通过本地服务器获取图片URL"""
        try:
            if not os.path.exists(self.image_dir):
                return None
            
            jpg_files = glob.glob(os.path.join(self.image_dir, "*.jpg"))
            png_files = glob.glob(os.path.join(self.image_dir, "*.png"))
            all_files = jpg_files + png_files
            
            if not all_files:
                return None
            
            latest_file = max(all_files, key=os.path.getmtime)
            filename = os.path.basename(latest_file)
            
            # 使用本地服务器URL
            image_url = f"{self.local_server_url}/{filename}"
            
            self.get_logger().info(f"使用图片URL: {image_url}")
            return image_url
            
        except Exception as e:
            self.get_logger().error(f"获取图片URL失败: {e}")
            return None

    def call_coze_api_with_image(self, question_text, image_base64):
        """调用Coze API进行图文问答 - 使用base64图片数据"""
        url = "https://api.coze.cn/v3/chat"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建图文问答消息内容 - 使用base64
        message_content = [
            {
                "type": "image",
                "image": image_base64  # 直接使用base64数据
            },
            {
                "type": "text", 
                "text": question_text
            }
        ]
        
        payload = {
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "stream": True,
            "auto_save_history": True,
            "additional_messages": [
                {
                    "role": "user",
                    "content": json.dumps(message_content, ensure_ascii=False),
                    "content_type": "object_string"
                }
            ]
        }
        
        try:
            self.get_logger().info("正在调用Coze API进行图文问答...")
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
            
            # 去除重复内容
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

    def call_coze_api_with_image_url(self, question_text, image_url):
        """备选方案：使用图片URL调用Coze API"""
        url = "https://api.coze.cn/v3/chat"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用图片URL的方案
        message_content = [
            {
                "type": "image",
                "file_url": image_url  # 使用图片URL
            },
            {
                "type": "text", 
                "text": question_text
            }
        ]
        
        payload = {
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "stream": True,
            "auto_save_history": True,
            "additional_messages": [
                {
                    "role": "user",
                    "content": json.dumps(message_content, ensure_ascii=False),
                    "content_type": "object_string"
                }
            ]
        }
        
        try:
            self.get_logger().info("正在调用Coze API（使用图片URL）...")
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            response.raise_for_status()
            
            full_reply = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data:'):
                        data_str = line_str[5:].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'content' in data and isinstance(data['content'], str):
                                content = data['content']
                                full_reply += content
                        except json.JSONDecodeError:
                            continue
            
            cleaned_reply = self.remove_duplicates(full_reply)
            return cleaned_reply.strip()
            
        except Exception as e:
            self.get_logger().error(f"API调用失败: {e}")
            return "抱歉，服务暂时不可用。"

    def remove_duplicates(self, text):
        """去除文本中的重复内容"""
        if not text:
            return text
            
        if len(text) < 20:
            return text
            
        half_length = len(text) // 2
        first_half = text[:half_length]
        second_half = text[half_length:]
        
        if first_half in second_half and len(first_half) > 10:
            self.get_logger().info("检测到重复内容，已去重")
            return first_half.strip()
        
        return text

    def save_conversation(self, user_text, bot_reply, image_url=None):
        """保存完整对话记录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/conversation_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"对话时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                f.write(f"用户问题: {user_text}\n")
                if image_url:
                    f.write(f"分析图片: {image_url}\n")
                else:
                    f.write("图片数据: base64编码\n")
                f.write("-" * 30 + "\n")
                f.write(f"AI回复: {bot_reply}\n")
                f.write("=" * 50 + "\n")
            
            self.get_logger().info(f"完整对话已保存: {filename}")
            
        except Exception as e:
            self.get_logger().error(f"保存对话失败: {e}")

    def save_coze_reply(self, bot_reply):
        """单独保存Coze回复到文档"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        detail_filename = f"{self.reply_dir}/coze_reply_{timestamp}.txt"
        latest_filename = f"{self.reply_dir}/latest_coze_reply.txt"
        
        try:
            with open(detail_filename, 'w', encoding='utf-8') as f:
                f.write(f"回复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 40 + "\n")
                f.write(f"{bot_reply}\n")
                f.write("=" * 40 + "\n")
            
            with open(latest_filename, 'w', encoding='utf-8') as f:
                f.write(bot_reply)
            
            self.get_logger().info(f"Coze回复已保存")
            
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
