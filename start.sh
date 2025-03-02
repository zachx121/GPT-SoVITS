#!/bin/bash

# 初始化 conda
source /root/miniconda3/etc/profile.d/conda.sh
# 激活 conda 环境
conda activate GPTSoVits

# 杀掉已经运行的 Python 进程
pkill -f '/root/GPT-SoVITS/service_GSV/GSV_server.py'
pkill -f '/root/miniconda3/envs/GPTSoVits/bin/python'

# 启动服务器并将输出重定向到 server.log
nohup python -m service_GSV.GSV_server 1 https://u212392-96ec-47a44b54.bjb1.seetacloud.com:8443  >/dev/null 2>&1 &

# 输出启动成功的消息
echo "GPT SoVITS server is starting..."


