#!/bin/bash

# 启动的消费者进程数量
TARGET_COUNT=$1

# 检查是否提供了目标进程数量
if [ -z "$TARGET_COUNT" ]; then
  echo "Usage: $0 <number_of_consumers>"
  exit 1
fi

# 查找并杀掉已有的 GSV_tran_consumer.py 进程
echo "Checking for existing $SCRIPT_NAME processes..."
EXISTING_PIDS=$(pgrep -f $SCRIPT_NAME)

if [ ! -z "$EXISTING_PIDS" ]; then
  echo "Found existing $SCRIPT_NAME processes: $EXISTING_PIDS"
  echo "Killing existing processes..."
  kill -9 $EXISTING_PIDS
  echo "Existing processes killed."
else
  echo "No existing $SCRIPT_NAME processes found."
fi

# 初始化 conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate GPTSoVits

# 启动指定数量的 GSV_tran_consumer.py 进程
echo "Starting $TARGET_COUNT instances of $SCRIPT_NAME..."
for ((i=1; i<=TARGET_COUNT; i++))
do
  nohup python -m service_GSV.GSV_train "$i" >/dev/null 2>&1 &

  echo "Started instance $i"
done

echo "All $TARGET_COUNT instances started."

# 提示用户查看日志
echo "Logs are automatically managed by the logging framework in your code."
echo "To view logs, check the log directory: ./logs"
echo "For example: tail -f ./logs/consumer-1.log"

