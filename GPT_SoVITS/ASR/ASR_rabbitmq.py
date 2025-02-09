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
from funasr import AutoModel

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
    path_asr = os.path.join(PROJ_DIR, 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    path_vad = os.path.join(PROJ_DIR, 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch')
    path_punc = os.path.join(PROJ_DIR, 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
    for fp in [path_asr, path_vad, path_punc]:
        assert os.path.exists(fp), f"ASR Model Not Exist: path='{fp}'"
    model = AutoModel(
        model=path_asr,
        model_revision="v2.0.4",
        vad_model=path_vad,
        vad_model_revision="v2.0.4",
        punc_model=path_punc,
        punc_model_revision="v2.0.4",
    )
    # print(model.generate("/Users/zhou/Downloads/手动切分0.wav"))
    # funasr v1.0.0 加载音频的逻辑：float32, 声道取单声道0，然后重采样到16khz
    # funasr v1.2.3 加载音频的逻辑：float32, 声道取均值，然后重采样到16khz
    # 所以直接用librosa来实现一下（看了下数据基本是一致的）
    # audio_arr, sr = librosa.load("/Users/zhou/Downloads/手动切分0.wav", sr=16000, mono=True)
    # print(model.generate(audio_arr, disable_pbar=True))

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
        if logger.level == logging.DEBUG:
            scipy.io.wavfile.write(f"./aa_{time.time():.2f}.wav", 16000, audio_arr_float32)
        # Audio(audio_arr_float32, rate=16000)
        logger.debug(f"received audio with duration: {audio_arr_float32.shape[0] / 16000:.2f}")
        res = model.generate(input=audio_arr_float32, disable_pbar=True)
        rsp = {"trace_id": param.get("traceId", ""),
               "text": res[0]['text']}
        rsp = json.dumps(rsp, ensure_ascii=False)  # ensure_ascii
        logger.debug(f" -asr rsp is '{rsp}")
        channel.basic_publish(exchange='',
                              routing_key=param["resultQueueName"],
                              body=rsp,
                              properties=pika.BasicProperties(content_type='application/json'))

    channel.close()
    connection.close()
    del model



