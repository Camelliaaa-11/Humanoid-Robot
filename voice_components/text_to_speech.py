#!/usr/bin/env python3
"""
文本转语音模块 - 与ASR解耦的独立TTS服务
订阅/coze_reply话题，将文本转换为语音并通过/xunfei/tts_play播放
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import os
import subprocess
import hashlib
import tempfile

class TextToSpeechNode(Node):
    def __init__(self):
        super().__init__('text_to_speech')
        
        # 订阅coze_processor的文本回复
        self.subscription = self.create_subscription(
            String,
            '/coze_reply',
            self.tts_callback,
            10
        )
        
        # 发布到讯飞播放接口
        self.play_publisher = self.create_publisher(
            String, 
            '/xunfei/tts_play', 
            10
        )
        
        # 语音缓存目录
        self.cache_dir = "/home/ubuntu/data/speech"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 探测可用的TTS工具
        self.available_tts_tools = self.detect_tts_tools()
        
        self.get_logger().info(f"TTS模块启动，可用工具: {self.available_tts_tools}")
        
    def detect_tts_tools(self):
        """探测系统中可用的TTS工具"""
        tools = []
        
        # 检查espeak
        try:
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode == 0:
                tools.append('espeak')
        except:
            pass
            
        # 检查pico2wave
        try:
            result = subprocess.run(['which', 'pico2wave'], capture_output=True, text=True)
            if result.returncode == 0:
                tools.append('pico2wave')
        except:
            pass
            
        # 检查ffmpeg（用于格式转换）
        try:
            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
            if result.returncode == 0:
                tools.append('ffmpeg')
        except:
            pass
            
        return tools
    
    def tts_callback(self, msg):
        """处理接收到的文本消息"""
        try:
            text = msg.data.strip()
            if not text:
                self.get_logger().warning("收到空文本消息")
                return
                
            self.get_logger().info(f"收到TTS文本: {text}")
            
            # 生成语音并播放
            success = self.text_to_speech(text)
            
            if success:
                self.get_logger().info("语音播放成功")
            else:
                self.get_logger().error("语音播放失败")
                
        except Exception as e:
            self.get_logger().error(f"处理TTS消息异常: {e}")
    
    def text_to_speech(self, text):
        """文本转语音主函数"""
        try:
            # 1. 生成缓存文件名
            audio_file = self.generate_audio_filename(text)
            
            # 2. 如果文件不存在，生成语音文件
            if not os.path.exists(audio_file):
                self.get_logger().info(f"生成语音文件: {audio_file}")
                if not self.generate_audio_file(text, audio_file):
                    return False
            
            # 3. 通过ROS 2播放
            return self.publish_play_command(audio_file)
            
        except Exception as e:
            self.get_logger().error(f"TTS处理异常: {e}")
            return False
    
    def generate_audio_filename(self, text):
        """根据文本生成唯一的文件名"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"tts_{text_hash}.wav")
    
    def generate_audio_file(self, text, output_file):
        """使用系统工具生成语音文件"""
        
        # 方案1: 使用espeak生成WAV
        if 'espeak' in self.available_tts_tools:
            try:
                # 清理文本中的特殊字符
                clean_text = text.replace('"', '').replace("'", "")
                cmd = f'espeak -v zh "{clean_text}" -w {output_file}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_file):
                    self.get_logger().info("使用espeak生成语音成功")
                    return True
                else:
                    self.get_logger().warning(f"espeak生成失败: {result.stderr}")
            except subprocess.TimeoutExpired:
                self.get_logger().warning("espeak执行超时")
            except Exception as e:
                self.get_logger().warning(f"espeak异常: {e}")
        
        # 方案2: 使用pico2wave
        if 'pico2wave' in self.available_tts_tools:
            try:
                cmd = f'pico2wave -l=zh-CN -w={output_file} "{text}"'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_file):
                    self.get_logger().info("使用pico2wave生成语音成功")
                    return True
            except Exception as e:
                self.get_logger().warning(f"pico2wave异常: {e}")
        
        # 方案3: 使用espeak通过stdout重定向
        if 'espeak' in self.available_tts_tools:
            try:
                cmd = f'espeak -v zh "{text}" --stdout > {output_file}'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(output_file):
                    self.get_logger().info("使用espeak stdout生成语音成功")
                    return True
            except Exception as e:
                self.get_logger().warning(f"espeak stdout异常: {e}")
        
        self.get_logger().error("所有TTS方案均失败")
        return False
    
    def publish_play_command(self, audio_file):
        """发布播放命令到讯飞接口"""
        if not os.path.exists(audio_file):
            self.get_logger().error(f"音频文件不存在: {audio_file}")
            return False
        
        try:
            play_cmd = {"file": audio_file}
            msg = String()
            msg.data = json.dumps(play_cmd)
            
            self.play_publisher.publish(msg)
            self.get_logger().info(f"发布播放命令: {os.path.basename(audio_file)}")
            return True
        except Exception as e:
            self.get_logger().error(f"发布播放命令失败: {e}")
            return False

def main():
    rclpy.init()
    tts_node = TextToSpeechNode()
    
    try:
        rclpy.spin(tts_node)
    except KeyboardInterrupt:
        pass
    finally:
        tts_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()