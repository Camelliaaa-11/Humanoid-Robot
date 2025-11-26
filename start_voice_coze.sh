#!/bin/bash
# start_voice_coze.sh - 启动完整语音交互系统

echo "启动人形机器人语音交互系统..."

# 检查ROS2环境
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✅ ROS2环境已设置"
else
    echo "❌ 错误: 找不到ROS2环境，请先安装ROS2 Humble"
    exit 1
fi

# 进入项目目录
cd ~/voice_project

# 启动语音识别节点
echo "启动语音识别节点..."
python3 voice_to_text.py &
VOICE_PID=$!

sleep 3

# 启动Coze处理节点
echo "启动Coze处理节点..."
python3 coze_processor.py &
COZE_PID=$!

sleep 3

# 启动TTS转换节点
echo "启动TTS转换节点..."
python3 respond_to_mp3.py &
TTS_PID=$!

echo "等待系统启动..."
sleep 5

echo "系统启动完成!"
echo "语音识别PID: $VOICE_PID"
echo "Coze处理PID: $COZE_PID"
echo "TTS转换PID: $TTS_PID"
echo ""
echo "现在可以对机器人说话，系统会自动："
echo "1. 识别语音 → 2. 调用Coze → 3. 生成MP3回复"
echo "按 Ctrl+C 停止系统"

wait

kill $VOICE_PID $COZE_PID $TTS_PID 2>/dev/null
echo "系统已停止"
