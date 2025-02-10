import re
import sys

import numpy as np
import wave
import os
import logging
import base64
import json
import pika
import librosa
import torchaudio
import time
import gzip
from logging.handlers import TimedRotatingFileHandler
import multiprocessing as mp
import scipy
import soundfile
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from faster_whisper import WhisperModel
import torch

# PROJ_DIR = "/Users/zhou/0-Codes/GPT-SoVITS"
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../../"))
print(f"PROJ_DIR: {PROJ_DIR}")
INP_QUEUE = "queue_self_asr_request"


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        # "virtual_host": "device-public",
        "virtual_host": "test",
    }
    try:
        # 连接到 RabbitMQ
        credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
        parameters = pika.ConnectionParameters(
            host=rabbitmq_config["address"],
            port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
            virtual_host=rabbitmq_config["virtual_host"],
            credentials=credentials,
            connection_attempts=3,  # 最多尝试 3 次
            retry_delay=5,         # 每次重试间隔 5 秒
            socket_timeout=10      # 套接字超时时间为 10 秒
        )
        logger.info("mq配置完毕，开始blocking connect连接")
        connection = pika.BlockingConnection(parameters)
        logger.info("mq连接完毕，获取到connection")
        channel = connection.channel()
        logger.info("mq连接完毕，获取到chanel")

        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None


if __name__ == '__main__':
    # 获取实例编号（从启动参数中获取）
    instance_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    # 配置日志
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f"consumer-{instance_id}.log"),  # 日志文件路径
        when="midnight",  # 按天分隔（午夜生成新日志文件）
        interval=1,  # 每 1 天分隔一次
        backupCount=7,  # 最多保留最近 7 天的日志文件
        encoding="utf-8"  # 设置编码，避免中文日志乱码
    )
    file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
    file_handler.setFormatter(logging.Formatter(
        fmt='[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)

    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    dir_whisper = os.path.join(PROJ_DIR, 'tools/asr/models/faster_whisper_large_v3')
    assert os.path.exists(dir_whisper)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 直接下载的目录还无法加载，需要根据配置指定到哪一个blobl
    with open(os.path.join(dir_whisper, "refs", "main"), "r") as fr:
        _version = fr.readlines()[0]
        fp_whisper = os.path.join(dir_whisper, "snapshots", _version)
        assert os.path.exists(fp_whisper)
    model = WhisperModel(fp_whisper, device=device, compute_type="float32")

    if False:
        segments, info = model.transcribe(
            audio="/Users/zhou/Downloads/手动切分0.wav",
            beam_size=5,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=700),
            language=None)
        text = "".join([seg.text for seg in segments])
        print(f"text: '{text}'")

    if False:
        audio_arr, sr = librosa.load("/Users/zhou/Downloads/手动切分0.wav", sr=16000, mono=True)
        segments, info = model.transcribe(
            audio=audio_arr,
            beam_size=5,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=700),
            language=None)
        text = "".join([seg.text for seg in segments])
        print(f"text: '{text}'")

    connection, channel = connect_to_rabbitmq()
    logger.info("Connected. Start Consuming...")
    while True:
        method_frame, header_frame, body = channel.basic_get(queue=INP_QUEUE, auto_ack=True)
        if body is None:
            time.sleep(0.1)  # 如果没有消息，休眠一段时间
            continue  # 如果没有消息，等待下一次
        param = json.loads(body.decode('utf-8'))
        audio_bytes = gzip.decompress(base64.b64decode(param["gzipAudio"]))
        audio_arr_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_arr_float32 = audio_arr_int16.astype(np.float32) / 32768.0
        # scipy.io.wavfile.write(f"./aa_{time.time():.2f}.wav", 16000, audio_arr_float32)
        # Audio(audio_arr_float32, rate=16000)
        logger.debug(f"received audio with duration: {audio_arr_float32.shape[0] / 16000:.2f}")
        segments, info = model.transcribe(
            audio=audio_arr,
            beam_size=5,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=700),
            language=None)
        text = "".join([seg.text for seg in segments])
        rsp = {"trace_id": param.get("traceId", ""),
               "text": text}
        rsp = json.dumps(rsp, ensure_ascii=False)  # ensure_ascii
        logger.debug(f" -asr rsp is '{rsp}")
        channel.basic_publish(exchange='',
                              routing_key=param["resultQueueName"],
                              body=rsp,
                              properties=pika.BasicProperties(content_type='application/json'))

    channel.close()
    connection.close()
    del model



