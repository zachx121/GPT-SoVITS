import logging
from logging.handlers import TimedRotatingFileHandler

import os
import shutil
import sys
import json
import yaml
import wave
import re
import time
import base64
import socket
import traceback
import multiprocessing as mp

from urllib.parse import unquote
from subprocess import Popen, getstatusoutput
from threading import Thread

import requests as http_requests  # 给 requests 库设置别名

import utils_audio
from . import GSV_const as C
from .GSV_const import Route as R
import pika
import subprocess
assert getstatusoutput("ls tools")[0] == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python GPT_SoVITS/GSV_train.py"

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 人声分离
# cmd = "demucs --two-stems=vocals xx"

queue_tran_request = "queue_tran_request"
queue_tran_result = "queue_tran_result"


# 创建流读取函数
def log_stream(stream, logger, level=logging.INFO):
    while True:
        line = stream.readline()
        if not line:
            break
        if line.strip():
            logger.log(level, line.strip())


def train_consumer():
    connection, channel = connect_to_rabbitmq()

    while True:
        try:
            try:
                if channel is None or channel.is_closed:
                    connection, channel = connect_to_rabbitmq()
            except Exception:
                # 如果 抛出异常，直接重连
                logger.info(f"mq connect error,reconnect")
                connection, channel = connect_to_rabbitmq()
    
            method_frame, header_frame, body = channel.basic_get(queue=queue_tran_request, auto_ack=True)
            if body is None:
                time.sleep(0.1)  # 如果没有消息，休眠一段时间
                continue  # 如果没有消息，等待下一次
            # 解析任务消息
            task = json.loads(body)
            #logger.info(f"Received training task: {task}")

            # 执行训练逻辑
            #train_model(task)
            
            #1.0.10.3 更新v2模型，方法改成直接用python命令调用脚本，如果脚本没有异常则发送训练成功消息，如果有异常则发送训练失败消息
            # 执行训练逻辑
            sid = task['speaker']
            LANG = task['lang']
            data_urls = task['data_urls']
            # 只取 data_urls 的第一个元素
            first_data_url = str(data_urls[0])  # 确保是字符串类型
            try:
                # 调用训练脚本
                logger.info(f">>> speaker='{sid}', Language='{LANG}'")
                logger.info(f">>> data_urls: {first_data_url}")
               # 启动子进程（添加 -u 参数禁用缓冲）
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "service_GSV.GSV_train_standalone", sid, LANG, first_data_url],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # 行缓冲
                    universal_newlines=True,
                    encoding='utf-8'  # 指定编码
                )

                # 创建并启动日志线程
                stdout_thread = Thread(target=log_stream, args=(proc.stdout, logger, logging.INFO))
                stderr_thread = Thread(target=log_stream, args=(proc.stderr, logger, logging.ERROR))
                stdout_thread.start()
                stderr_thread.start()

                # 等待进程结束
                return_code = proc.wait()

                # 等待日志线程结束
                stdout_thread.join()
                stderr_thread.join()

                if  return_code == 0:
                    logger.info("Training script executed successfully")
                    result= {"code": 0, "msg": "Model Training finish.", "result": sid}
                    # 直接重新建立连接，因为tran耗时比较久，这时候断开了
                    connection, channel = connect_to_rabbitmq()
                    # 发送结果到队列
                    send_result_with_retry(channel, result)
                elif return_code == 502:
                    logger.error(f"样本数量异常 ({sid})")
                    result = {"code": 502, "msg": "样本数量异常", "result": task.get('speaker', 'unknown')}
                    # 发送结果到队列
                    send_result_with_retry(channel, result)
                else:
                    logger.error(f"Training script failed with error")
                    result = {"code": 1, "msg": "Model Training failed.", "result": task.get('speaker', 'unknown')}
                    # 发送结果到队列
                    send_result_with_retry(channel, result)
            except Exception as e:
                logger.error(f"Error during training script execution: {e}", exc_info=True)            


        except pika.exceptions.AMQPConnectionError:
            logger.error("Connection to RabbitMQ lost, attempting to reconnect...")
            connection, channel = connect_to_rabbitmq()
        except Exception as e:
            logger.error(f"Error in task: {e}", exc_info=True)
            result = {"code": 1, "msg": "Model Training failed.", "result": task.get('speaker', 'unknown')}
            # 发送结果到队列
            send_result_with_retry(channel, result)
    

def send_result_with_retry(channel, result):
    for attempt in range(5):
        try:
            channel.basic_publish(
                exchange='',
                routing_key=queue_tran_result,
                body=json.dumps(result),
                properties=PROPERTIES  # Make message persistent
            )
            logger.info("send message to queue")

            break  # 发送成功，退出循环
        except pika.exceptions.ChannelClosed:
            logger.error("Channel closed, reconnecting...")
            connection, channel = connect_to_rabbitmq()  # 重新连接
        except Exception as e:
            logger.error(f"Failed to send result: {e}")
            time.sleep(2)  # 等待后重试


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        "virtual_host": "test-0208"
    }

    # 连接到 RabbitMQ
    credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config["address"],
        port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
        virtual_host=rabbitmq_config["virtual_host"],
        credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None
    

if __name__ == "__main__":
    try:
        # 获取实例编号（从启动参数中获取）
        if len(sys.argv) > 1:
            instance_id = sys.argv[1]  # 启动时传入的实例编号
        else:
            instance_id = "default"  # 如果未传入参数，使用默认值

        # 日志目录（代码中自动创建，无需脚本管理）
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        else:
            print(f"Log directory already exists: {log_dir}")

        # 根据实例编号生成日志文件名
        log_file = f"consumer-{instance_id}.log"
        log_path = os.path.join(log_dir, log_file)

        file_handler = TimedRotatingFileHandler(
        filename=log_path,  # 日志文件路径
        when="midnight",    # 按天分隔（午夜生成新日志文件）
        interval=1,         # 每 1 天分隔一次
        backupCount=7,      # 最多保留最近 7 天的日志文件
        encoding="utf-8"    # 设置编码，避免中文日志乱码
        )
        file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
        file_handler.setFormatter(logging.Formatter(
            fmt='[%(asctime)s-%(levelname)s]: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # 将文件处理器添加到日志记录器中
        logger.addHandler(file_handler)

        train_consumer()
    except KeyboardInterrupt:
        logger.info("Consumer stopped.")