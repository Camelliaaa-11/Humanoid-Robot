#!/bin/bash
# start_complete_system.sh - 启动完整语音交互系统

echo "启动人形机器人语音交互系统..."

# 检查ROS2环境
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✅ ROS2环境已设置"
else
    echo "❌ 错误: 找不到ROS2环境，请先安装ROS2 Humble"
    exit 1
fi

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "✅ 工作目录: $SCRIPT_DIR"

# 创建图片目录
echo "创建图片目录..."
mkdir -p ~/rviz_captured_images

sleep 2

# 启动HTTP服务器（为图片提供服务）
echo "启动HTTP图片服务器..."
cd ~/rviz_captured_images
python3 -m http.server 8080 &
HTTP_SERVER_PID=$!
echo "HTTP服务器PID: $HTTP_SERVER_PID"

sleep 2

# 启动ngrok隧道
echo "启动ngrok隧道..."
ngrok http 4040 &
NGROK_PID=$!
echo "ngrok PID: $NGROK_PID"

sleep 5  # 给ngrok更多时间启动

# 回到脚本目录
cd "$SCRIPT_DIR"

# 启动图片捕捉节点
echo "启动RViz2图片捕捉节点..."
python3 direct_capture_node.py &
IMAGE_CAPTURE_PID=$!
echo "图片捕捉PID: $IMAGE_CAPTURE_PID"

sleep 5

# 启动语音识别节点
echo "启动语音识别节点..."
python3 voice_to_text.py &
VOICE_PID=$!

sleep 3

# 启动Coze处理节点
echo "启动Coze处理节点..."
python3 coze_processor_textandimage.py &
COZE_PID=$!

sleep 3

# 启动TTS转换节点
echo "启动TTS转换节点..."
python3 text_to_speech.py &
TTS_PID=$!

echo "等待系统启动..."
sleep 8  # 增加等待时间确保所有服务就绪

echo "系统启动完成!"
echo "HTTP服务器PID: $HTTP_SERVER_PID"
echo "ngrok PID: $NGROK_PID" 
echo "图片捕捉PID: $IMAGE_CAPTURE_PID"
echo "语音识别PID: $VOICE_PID"
echo "Coze处理PID: $COZE_PID"
echo "TTS转换PID: $TTS_PID"
echo ""
echo "系统启动顺序："
echo "1. HTTP服务器 → 2. ngrok隧道 → 3. 图片捕捉 → 4. 语音识别 → 5. Coze处理 → 6. TTS转换"
echo ""
echo "现在可以对机器人说话，系统会自动："
echo "1. 捕捉RViz2图像 → 2. 识别语音 → 3. 调用Coze进行图文问答 → 4. 生成语音回复"
echo "按 Ctrl+C 停止系统"

# 等待用户中断
trap 'kill $HTTP_SERVER_PID $NGROK_PID $IMAGE_CAPTURE_PID $VOICE_PID $COZE_PID $TTS_PID 2>/dev/null; echo "系统已停止"; exit' INT

wait
