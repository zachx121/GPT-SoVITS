#!/bin/bash
set -e

# 定义服务的端口范围
START_PORT=7001
END_PORT=7010

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 启动服务
for PORT in $(seq $START_PORT $END_PORT); do
    LOG_FILE="$LOG_DIR/service_ASR_$PORT.log"
    nohup python service_ASR/ASR_http.py "$PORT" > "$LOG_FILE" 2>&1 &
    echo "Started service on port $PORT, logging to $LOG_FILE"
done

echo "All services started."