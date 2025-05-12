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
import torch
import soundfile
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from funasr import AutoModel
from faster_whisper import WhisperModel

# PROJ_DIR = "/Users/zhou/0-Codes/GPT-SoVITS"
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))
print(f"PROJ_DIR: {PROJ_DIR}")
INP_QUEUE = "queue_service_self_asr_request"


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        # "virtual_host": "device-public",
        "virtual_host": "test-0208",
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


class ASRModelWrapper:
    def __init__(self, lang):
        assert lang in ["zh_cn", "en_us"]
        self.lang = lang
        self.model = self.load_zh_asr_model() if self.lang=="zh_cn" else self.load_other_asr_model()

    def predict(self, inp):
        # inp: audio_arr_float32 or file_path
        if self.lang == "zh_cn":
            # res: [{'key': '董宇辉带货_16k_mono', 'text': '那会儿人们对于沟通是有多么强烈的欲望呀，啊就像那些痛苦时刻的诗人有那么强烈的。'}]
            res = self.model.generate(input=inp, disable_pbar=True)
            if res is not None and len(res) > 0:
                asr_txt = res[0]['text']
            else:
                asr_txt = ""
            return asr_txt
        else:
            segments, info = self.model.transcribe(
                audio=inp,
                beam_size=5,
                # vad_filter=True,
                # vad_parameters=dict(min_silence_duration_ms=700),
                language=None)
            asr_txt = "".join([seg.text for seg in segments])
            return asr_txt

    @staticmethod
    def load_zh_asr_model():
        logger.info(f">>> loading asr model with [damo]")
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
            # vad_model=path_vad,
            # vad_model_revision="v2.0.4",
            punc_model=path_punc,
            punc_model_revision="v2.0.4",
            disable_update=True
        )
        return model

    @staticmethod
    def load_other_asr_model():
        logger.info(f">>> loading asr model with [whisper]")
        # paraformer-zh is a multi-functional asr model
        # use vad, punc, spk or not as you need
        dir_whisper = os.path.join(PROJ_DIR, 'tools/asr/models/faster-whisper-large-v3-local')
        assert os.path.exists(dir_whisper)
        # 直接下载的目录还无法加载，需要根据配置指定到哪一个blobl
        with open(os.path.join(dir_whisper, "refs", "main"), "r") as fr:
            _version = fr.readlines()[0]
            fp_whisper = os.path.join(dir_whisper, "snapshots", _version)
            assert os.path.exists(fp_whisper)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = WhisperModel(fp_whisper, device='cuda', compute_type="float16")
        return model


# python service_ASR/ASR_rabbitmq_zh.py 'zh_cn' 4G*5
# python service_ASR/ASR_rabbitmq_zh.py 'en_us' 6G*3
if __name__ == '__main__':
    # 获取实例编号（从启动参数中获取）
    lang = sys.argv[1] if len(sys.argv) > 1 else "zh_cn"
    instance_id = sys.argv[2] if len(sys.argv) > 2 else "default"
    log_dir = "logs"
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

    model = ASRModelWrapper(lang=lang)
    # warm-up
    for _ in range(5):
        res = model.predict(inp="/root/autodl-fs/audio_samples/董宇辉带货_16k_mono.wav")
        logger.debug(f"warm-up inference. asr result:{res}")
    # print(model.generate("/Users/zhou/Downloads/手动切分0.wav"))
    # funasr v1.0.0 加载音频的逻辑：float32, 声道取单声道0，然后重采样到16khz
    # funasr v1.2.3 加载音频的逻辑：float32, 声道取均值，然后重采样到16khz
    # 所以直接用librosa来实现一下（看了下数据基本是一致的）
    # audio_arr, sr = librosa.load("/Users/zhou/Downloads/手动切分0.wav", sr=16000, mono=True)
    # print(model.generate(audio_arr, disable_pbar=True))

    connection, channel = connect_to_rabbitmq()

    def call_back_func(ch, method, properties, body):
        print(f">>> basic_get收到消息")
        try:
            stime = time.time()
            param = json.loads(body.decode('utf-8'))
            tid = param.get("trace_id", "")
            _stime = time.time()
            logger.debug(f">>> 收到消息，距离tid时间戳: {stime*1000 - int(tid.split('_')[0]):.0f}ms (stime:{stime*1000:.0f} tid:{int(tid.split('_')[0])})")
            logger.debug(f">>> json解析耗时: {(_stime - stime) * 1000:.0f}ms")
            logger.debug(f">>> 收到消息且解析后，距离tid时间戳: {_stime*1000 - int(tid.split('_')[0]):.0f}ms")
            opt_queue = param["result_queue_name"]
            audio_bytes = base64.b64decode(param["audio_buffer"])
            try:
                audio_arr_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_arr_float32 = audio_arr_int16.astype(np.float32) / 32768.0
                # scipy.io.wavfile.write(f"./aa_{time.time():.2f}.wav", 16000, audio_arr_float32)
                # Audio(audio_arr_float32, rate=16000)
                logger.debug(f"received audio with duration: {audio_arr_float32.shape[0] / 16000:.2f}")
                asr_txt = model.predict(inp=audio_arr_float32)
                logger.debug(f">>> 推理完成 elapse:{(time.time() - stime) * 1000:.0f}ms tid:{tid} opt_queue:{opt_queue} asr:{asr_txt}")
                rsp = {"trace_id": tid,
                       "audio_text": asr_txt,
                       "finish_time": int(time.time()*1000)}
                rsp = {"code": 0,
                       "msg": "",
                       "result": rsp}
                rsp = json.dumps(rsp, ensure_ascii=False)  # ensure_ascii
                logger.debug(f" -asr rsp is '{rsp}")
                channel.basic_publish(exchange='',
                                      routing_key=opt_queue,
                                      body=rsp,
                                      properties=pika.BasicProperties(content_type='application/json'))
                logger.debug(f">> 推理完成且发回队列 elapse:{(time.time() - stime)*1000:.0f}ms")
                if logger.level == logging.DEBUG:
                    scipy.io.wavfile.write(f"./asr_debug_{tid}.wav", 16000, audio_arr_float32)
            except Exception as e:
                rsp = {"code": 1,
                       "msg": f"ASR Service Error. {e}",
                       "result": {}}
                rsp = json.dumps(rsp, ensure_ascii=False)  # ensure_ascii
                channel.basic_publish(exchange='',
                                      routing_key=opt_queue,
                                      body=rsp,
                                      properties=pika.BasicProperties(content_type='application/json'))
        except Exception as e:
            logger.error("Param Parse Error.")

    logger.info("Connected. Start Consuming...")
    channel.basic_consume(queue=INP_QUEUE, auto_ack=True, on_message_callback=call_back_func)
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error(f"Error during consuming error: {e}")
        raise e
    finally:
        # 关闭通道和连接
        logger.warning(f"close channel/connection and del M in try-catch...")
        channel.close()
        connection.close()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

