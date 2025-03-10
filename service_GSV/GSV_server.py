"""
[route]: /train_model
[json-params]: {speaker:"..", lang:"zh", data_urls:["...", "..." ]}

[route]: /model_status
[json-params]: {speaker_list: ["..", ".."]}


[route]: /add_reference
[json-params]:
{"speaker": "..",
 "ref_audio_url": "...",
 "ref_text_url": "...", 注意文本格式文本是 "zh_cn|你好我是四郎" 这样的，即按竖线分割的，前面是语言后面是音频的文本
 "ref_suffix":".." optional 可以直接不传，传了的话后面推理接口也要传。当可以提供多个参考音频时，可以用情绪作为后缀}

[route]: /init_model
[json-params]: {speaker:"..."}

[route]: /inference
[json-params]:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. "JP", "ZH", "EN", "ZH_EN", "JP_EN")
    use_ref: bool = True  # 推理时是否使用参考音频的情绪
    ref_suffix: str = D_REF_SUFFIX  # 可不传，用于指定参考音频的后缀
"""

import re
import sys
import librosa
import torch
import time
import numpy as np
import os
os.environ['TQDM_DISABLE'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
from logging.handlers import TimedRotatingFileHandler
import base64
import json
import shutil
from urllib.parse import unquote
from subprocess import getstatusoutput, check_output
from flask import Flask, request

from GSV_model import GSVModel, ReferenceInfo
import GSV_const as C
from GSV_const import Route as R
import multiprocessing as mp
import utils_audio
import socket
import pika
import queue
import traceback  # 导入 traceback 模块
import schedule
import threading
import atexit
import pyloudnorm as pyln

app = Flask(__name__, static_folder="./static_folder", static_url_path="")

# 日志目录
log_dir = "../GPT_SoVITS/logs"
# 日志文件名
log_file = "server.log"
queue_service_inference_request_prefix='queue_service_inference_request_'


def get_machine_id():
    """获取机器的主机名，清理并返回。"""
    machine_id = None
    try:
        # 尝试通过 socket 获取主机名
        machine_id = socket.gethostname()
        
        # 如果获取失败，尝试通过环境变量获取
        if not machine_id:
            machine_id = os.getenv("HOSTNAME")  # Linux/Docker
        if not machine_id:
            machine_id = os.getenv("COMPUTERNAME")  # Windows

        # 清理主机名：只保留字母、数字和横线
        if machine_id:
            machine_id = re.sub(r'[^a-zA-Z0-9\-]', '', machine_id).lower()
        logger.info("get  machine ID: %s", machine_id)
        return machine_id

    except Exception as e:
        logger.error("Error getting machine ID: %s", repr(e))
        return None


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        "virtual_host": "device-public"
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


def adjust_loudness(audio_arr, target_lufs=-23.0):
    """
    调整音频的响度（LUFS）到目标响度（LUFS）。
    :param audio_arr: 输入音频（[-1.0, 1.0] 范围的 numpy 数组）
    :param target_lufs: 目标 LUFS（默认为 -23 LUFS）
    :return: 调整后的音频
    """
    # 计算当前音频的 LUFS
    meter = pyln.Meter(16000)  # 采样率是 16kHz
    current_lufs = meter.integrated_loudness(audio_arr)

    # 计算增益
    gain = target_lufs - current_lufs

    # 根据增益调整音频
    adjusted_audio = pyln.normalize.loudness(audio_arr, current_lufs, target_lufs)
    
    return adjusted_audio


def model_process(sid: str, event, q_inp):
    global logger
    exchange_service_load_model_result='exchange_service_load_model_result'

    logger=config_log()
    # 获取机器的 hostname
    logger.info("开始运行model_process")
    machine_id = get_machine_id()
    if not machine_id:
        logger.error("Failed to retrieve machine ID.")
        return
    logger.info("检测模型文件是否存在")
    # 从OSS上加载模型
    if not (os.path.exists(R.get_sovits_fp(sid)) and os.path.exists(R.get_gpt_fp(sid))):
        try:
            logger.info(f"download oss models of '{sid}'")
            logger.info(f"- sovits_fp: {R.get_sovits_fp(sid)}")
            logger.info(f"- gpt_fp: {R.get_gpt_fp(sid)}")
            utils_audio.download_from_qiniu(R.get_sovits_osskey(sid), R.get_sovits_fp(sid))
            utils_audio.download_from_qiniu(R.get_gpt_osskey(sid), R.get_gpt_fp(sid))
            logger.info("download finished")
        except Exception as e:
            res = {"code": 0, "msg": "", "result": ""}
            logger.error(f"error when download '{sid}': {repr(e)}")
            res['code'] = 1
            res['msg'] = f"model of '{sid}' is not found and download failed"
            return json.dumps(res)

    # 下载完成后检查文件是否存在
    if not os.path.exists(R.get_sovits_fp(sid)) or not os.path.exists(R.get_gpt_fp(sid)):
        # 如果文件未下载成功，发送load失败事件
        logger.error(f"Model files for '{sid}' not found after download.")
        load_result_event = {
            "uniqueVoiceName": sid,  # 唯一语音名称
            "loadStatus": False,      # 加载失败
            "error": "Model download failed or file missing"
        }
        # 发送模型加载失败事件到 MQ
        connection, channel = connect_to_rabbitmq()
        if connection and channel:
            channel.basic_publish(
                exchange=exchange_service_load_model_result, 
                routing_key='', 
                body=json.dumps(load_result_event),  
                properties=PROPERTIES
            )
        return  # 退出函数
    

    
    request_queue_name = queue_service_inference_request_prefix+sid
    

    # 连接到 RabbitMQ
    connection, channel = connect_to_rabbitmq()
    if not connection or not channel:
        logger.info("连接mq失败")
        return  # 如果连接失败，退出函数
    # 设置过期时间（单位：毫秒），这里设置为 1 小时（3600000 毫秒）
    args = {
        'x-expires': 60 * 60 * 1000  # 设置过期时间
    }
    # 声明队列并设置参数
    channel.queue_declare(queue=request_queue_name, 
                          durable=False,    # 队列是否持久化
                          exclusive=False,  # 是否为独占队列
                          auto_delete=False, # 是否自动删除
                          arguments=args)   # 额外的参数（如过期时间）
    
    M = GSVModel(sovits_model_fp=R.get_sovits_fp(sid),
                 gpt_model_fp=R.get_gpt_fp(sid),
                 speaker=sid)
    
    # 发送load成功事件
    logger.info("发送load成功事件到mq")
    load_result_event = {
        "uniqueVoiceName": sid,  # 唯一语音名称
        "loadStatus":True
    }
    channel.basic_publish(exchange=exchange_service_load_model_result, routing_key='', body=json.dumps(load_result_event),  properties=PROPERTIES)
    logger.info("event设置为set")
    event.set()
    

    while True:
      
        # 进程是否关闭逻辑
        try:
            # 非阻塞地从队列获取参数
            p = q_inp.get_nowait()  # 立即返回，不阻塞
        except queue.Empty:
            p = None  # 如果队列为空，设置 p 为 None

        # 检查 p 是否为关闭信号
        if p == "STOP":  # 使用字符串 "STOP" 作为关闭信号
            logger.info(f"Received shutdown signal for SID: {sid}. Exiting process.")
            break  # 接收到关闭信号，退出循环
            
            
        # 从 RabbitMQ 获取消息
        try:
            method_frame, header_frame, body = channel.basic_get(queue=request_queue_name, auto_ack=True)
            if body is None:
                time.sleep(0.1)  # 如果没有消息，休眠一段时间
                continue  # 如果没有消息，等待下一次
            # 尝试解析 JSON
            # 将字节串解码为字符串
            body_str = body.decode('utf-8')
            # 将字符串解析为字典
            info_dict = json.loads(body_str)
            # 创建 InferenceParam 实例
            p = C.InferenceParam(info_dict)
            # 处理 InferenceParam 实例
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误: {e}")
            logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
            continue;
        except Exception as e:
            logger.error(f"创建 InferenceParam 实例时出错: {e}")
            logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
            continue  # 继续下一个循环

        try:
            # 检查是否设置过默认的ref
            ref_audio_fp = R.get_ref_audio_fp(p.speaker, C.D_REF_SUFFIX)
            ref_text_fp = R.get_ref_text_fp(p.speaker, C.D_REF_SUFFIX)
            if not (os.path.exists(ref_audio_fp) and os.path.exists(ref_text_fp)):
                logger.warning(f">>> reference as '{p.ref_suffix}' is not ready, init a default one from training data.")
                add_default_ref(p.speaker)

            tlist = []
            tlist.append(int(time.time() * 1000))
            if p.debug:
                logger.info(f"""params as: 
                target_text={p.text},
                target_lang={p.tgt_lang},
                ref_info={p.ref_info}, # ReferenceInfo(audio_fp={p.ref_info.audio_fp},text={p.ref_info.text},lang={p.ref_info.lang}) 
                top_k=1, top_p=0, temperature=0,
                ref_free={p.ref_free}, no_cut={p.nocut}
                """)
            wav_sr, wav_arr_int16 = M.predict(target_text=p.text,
                                              target_lang=p.tgt_lang,
                                              ref_info=p.ref_info,
                                              top_k=20, top_p=1.0, temperature=0.99,
                                              ref_free=p.ref_free, no_cut=p.nocut)
            if p.debug:
                # 后处理之前的音频
                import soundfile as sf
                sf.write(f"{sid}_{time.time():.0f}_ori.wav", wav_arr_int16, wav_sr)
                tlist.append(int(time.time()*1000))
            
           
            # 后处理 | (int16,random_sr)-->(int16,16khz)
            wav_arr_float32 = wav_arr_int16.astype(np.float32) / 32768.0
            wav_arr_float32_16khz = librosa.resample(wav_arr_float32, orig_sr=wav_sr, target_sr=16000)
             # 调整响度
            wav_arr = adjust_loudness(wav_arr_float32_16khz, target_lufs=-23.0)  # 假设目标响度为 -23 LUFS
            wav_arr_int16 = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
            if p.debug:
                sf.write(f"{sid}_{time.time():.0f}_16khz.wav", wav_arr_int16, 16000)
            tlist.append(int(time.time()*1000))
            rsp = {"trace_id": p.trace_id,
                   # "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
                   "audio_buffer_int16": base64.b64encode(wav_arr_int16.tobytes()).decode(),
                   "sample_rate": 16000,
                   "audio_text": p.text
                  }
            rsp = json.dumps({"code": 0,
                              "msg": "",
                              "result": rsp})

            channel.basic_publish(exchange='', routing_key=p.result_queue_name, body=rsp,  properties=PROPERTIES)
            tlist.append(int(time.time()*1000))

        except Exception as e:
            logger.error(f">>> Error when model.predict. e: {repr(e)}")
            logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪

            rsp = {"trace_id": p.trace_id,
                   "audio_buffer_int16": "",
                   "sample_rate": 16000}
            rsp = json.dumps({"code": 1,
                              "msg": f"Prediction failed, internal err {repr(e)}",
                              "result": rsp})
            channel.basic_publish(exchange='', routing_key=p.result_queue_name, body=rsp,  properties=PROPERTIES)

    # 清理资源
    channel.close()
    connection.close()
    del M
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# 返回所有GPU的内存空余量，是一个list
def get_free_gpu_mem():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


@app.route("/load_model", methods=['POST'])
def load_model():
    """
    http params:
    - speaker:str
    - speaker_num:int
    """
    res = {"code": 0, "msg": "", "result": ""}
    info = request.get_json()
    sid = info['speaker']
    sid_num = info['speaker_num']
#   download_overwrite = info.get("download_overwrite", "0")
    logger.info(f"load_model: {info}")

    if sid in M_dict:
        # 直接全部unload重新加载，减少逻辑
        cur_num = len(M_dict[sid]['process_list'])
        res['code'] = 1
        res['msg'] = f"Already Init ({sid}_x{cur_num}). call `unload_model` first."
        return json.dumps(res)

    # 假设一个模型占用2.8GB显存
    if get_free_gpu_mem()[0] <= 2800 * (sid_num + 1):
        res['code'] = 1
        res['msg'] = 'GPU OOM'
        return json.dumps(res)


    # 开启N个子进程加载模型
    # 保留inp用于发布关闭信息
    q_inp = mp.Queue()
    process_list = []
    _load_events = []
    for _ in range(sid_num):
        event = mp.Event()
        p = mp.Process(target=model_process, args=(sid,event,q_inp))  # 移除 q_out

        process_list.append(p)
        _load_events.append(event)
        p.start()

    # # 阻塞直到所有模型加载完毕
    # while not all([event.is_set() for event in _load_events]):
    #     pass

    M_dict[sid] = {"q_inp": q_inp, "process_list": process_list, "load_events":_load_events}
    res['msg'] = "Init Success"
    return json.dumps(res)


@app.route("/unload_model", methods=['POST'])
def unload_model():
    info = request.get_json()
    sid = info["speaker"]
    if sid not in M_dict:
        res = {"code": 1,
               "msg": f"sid('{sid}') not loaded yet",
               "result": ""}
        return res

    # 有N个子进程，所以给队列放入N个None确保每个子进程全都能收到结束信号
    for _ in range(len(M_dict[sid]["process_list"])):
        M_dict[sid]['q_inp'].put("STOP")
    for p in M_dict[sid]["process_list"]:
        p.join()
    # 清理掉字典里记录的kv
    del M_dict[sid]
    res = {"code": 0,
           "msg": "Model Unloaded",
           "result": ""}
    return json.dumps(res)


def add_default_ref(sid, tryOSS=True):
    os.makedirs(os.path.join(C.VOICE_SAMPLE_DIR, sid), exist_ok=True)
    audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    text_fp = R.get_ref_text_fp(sid, C.D_REF_SUFFIX)
    if tryOSS:
        logger.info("downloading default_ref from OSS")
        logger.info(f"- ref_audio: {R.get_ref_audio_osskey(sid)} --> {audio_fp}")
        logger.info(f"- ref_text: {R.get_ref_text_osskey(sid)} --> {text_fp}")
        utils_audio.download_from_qiniu(R.get_ref_audio_osskey(sid), audio_fp)
        utils_audio.download_from_qiniu(R.get_ref_text_osskey(sid), text_fp)
        logger.info("download finished")
        return None

    asr_fp = os.path.join(C.VOICE_SAMPLE_DIR, sid, "asr", "denoised.list")

    assert os.path.exists(asr_fp)
    with open(asr_fp, "r", encoding='utf-8') as fr:
        # 示例如下：
        # voice_sample/test_silang1636/denoised/ref_audio_default.wav_0000000000_0000127680.wav|denoised|ZH|真是脚带帽在头顶靴上下不？
        line = fr.readline().strip()

    infos = line.split("|")
    _audio_fp = infos[0]
    _lang = infos[2]
    _lang_map = {v: k for k, v in C.LANG_MAP.items()}
    _lang = _lang_map[_lang]
    _text = infos[3]

    cmd1 = f"cp {_audio_fp} {audio_fp}"
    s1, o1 = getstatusoutput(cmd1)
    assert s1 == 0, f"execution fail. [cmd] {cmd1} [output] {o1}"
    with open(text_fp, "w") as fpw:
        fpw.write(f"{_lang}|{_text}")
    # cmd2 = f"echo '{_lang}|{_text}' > {text_fp}"
    # s2, o2 = getstatusoutput(cmd2)
    # assert s2 == 0, f"execution fail. [cmd] {cmd2} [output] {o2}"


@app.route("/add_reference", methods=['POST'])
def add_reference():
    """
    如果只传speaker表示从数据集选第一个作为reference
    http params:
    - speaker:str 必传
    - ref_suffix:str optional
    - ref_audio_url:str optional
    - ref_text_url:str optional
    """
    info = request.get_json()
    logger.debug(info)
    sid = info['speaker']
    suffix = info.get("ref_suffix", C.D_REF_SUFFIX)
    os.makedirs(os.path.join(C.VOICE_SAMPLE_DIR, sid), exist_ok=True)
    audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    text_fp = R.get_ref_text_fp(sid, C.D_REF_SUFFIX)

    # 如果没有用默认的就说明有指定的音频，否则用训练集里的一句话
    if suffix != C.D_REF_SUFFIX:
        logger.info(f">>> will use **ASSIGNED** audio&text with suffix: '{suffix}'")
        cmd = f"wget \"{info['ref_audio_url']}\" -O {audio_fp}"
        logger.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"code": 1,
                              "msg": f"Reference Audio Download Wrong",
                              "result": ""})
            return res

        cmd = f"wget \"{info['ref_text_url']}\" -O {text_fp}"
        logger.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"code": 1,
                              "msg": f"Reference Text Download Wrong",
                              "result": ""})
            return res

        res = json.dumps({"code": 0,
                          "msg": "Reference added.",
                          "result": ""})
    else:
        add_default_ref(sid)
        res = json.dumps({"code": 0,
                          "msg": "Automatic Reference added.",
                          "result": ""})
    return res


@app.route("/check_speaker", methods=['POST'])
def check_speaker():
    """
    http params:
    - 以Param类的成员变量为准
    """
    info = request.get_json()
    if info is None:
        return json.dumps({"code": 1,
                        "msg": "format error",
                        "result": ""})

    logger.warning(f"inference at {time.time():.03f}s info:{info}")
    p = C.InferenceParam(info)
    # 检查speaker是否已经加载
    if p.speaker not in M_dict:
        return json.dumps({"code": 1,
                           "msg": f"inference使用的角色音({p.speaker})未被加载。已加载角色音: {M_dict.keys()}",
                           "result": ""})
    # 检查是否设置过默认的ref
    ref_audio_fp = R.get_ref_audio_fp(p.speaker, C.D_REF_SUFFIX)
    ref_text_fp = R.get_ref_text_fp(p.speaker, C.D_REF_SUFFIX)
    if not (os.path.exists(ref_audio_fp) and os.path.exists(ref_text_fp)):
        logger.warning(f">>> reference as '{p.ref_suffix}' is not ready, init a default one from training data.")
        add_default_ref(p.speaker)
    
    return json.dumps({"code": 0,
                        "msg": "speaker exist",
                        "result": ""})


@app.route("/train_model", methods=['POST'])
def train_model():
    """
    http params:
    - speaker: str
    - lang: str
    - data_urls: list[str]
    """
    info = request.get_json()
    logger.debug(info)
    sid = info['speaker']
    lang = info['lang']
    data_urls = info['data_urls']
    post2oss = "1"
    data_dir = os.path.join(C.VOICE_SAMPLE_DIR, sid)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    logger.info(f">>> Start Data Preparing.")
    for url in data_urls:
        logger.info(f">>> Downloading Sample from {url}")
        filename = os.path.basename(unquote(url).split("?")[0])
        cmd = f"wget \"{url}\" -O {os.path.join(data_dir,filename)}"
        logger.debug(cmd)
        status, output = getstatusoutput(cmd)
        if status != 0:
            logger.error(f"    Download fail. url is {url}")
    if len(os.listdir(data_dir)) == 0:
        res = json.dumps({"code": 1,
                          "msg": "All Audio url failed to download.",
                          "result": ""})
        return res
    logger.info(f">>> Start Model Training.")
    # nohup python GPT_SoVITS/GSV_train.py zh_cn ChatTTS_Voice_Clone_4_222rb2j voice_sample/ChatTTS_Voice_Clone_4_222rb2j > ChatTTS_Voice_Clone_4_222rb2j.train 2>&1 &
    # python GPT_SoVITS/GSV_train.py en_us ChatTTS_Voice_Clone_0_Mike_yvmz voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz 1 > ChatTTS_Voice_Clone_0_Mike_yvmz.train
    cmd = f"nohup python GPT_SoVITS/GSV_train.py {C.LANG_MAP[lang]} {sid} {data_dir} {post2oss} > {sid}.train 2>&1 &"
    logger.info(f"    cmd is {cmd}")
    status, output = getstatusoutput(cmd)
    if status != 0:
        logger.error(f"    Model training failed. error:\n{output}")
        res = json.dumps({"code": 1,
                          "msg": "Model Training Failed",
                          "result": ""})
    else:
        res = json.dumps({"code": 0,
                          "msg": "Model Training Started.",
                          "result": ""})
    return res


@app.route("/check_training_status", methods=['POST'])
def check_training_status():
    # 检查本机执行了几个模型训练
    cmd = "ps -ef | grep 'python GPT_SoVITS/GSV_train.py' | grep -v 'grep'"
    status, output = getstatusoutput(cmd)
    # 注意没有训练进程时，status是1，output是空串
    if output == "":
        res = json.dumps({"code": 0,
                          "msg": "success",
                          "result": []})
    else:
        assert status == 0, f"cmd execution failed. cmd:'{cmd}' output:'{output}'"
        # sid_list = [re.split("\s+", i)[10] for i in output.split("\n")]
        sid_list = [re.split(r"\s+", i)[10] for i in output.split("\n") if i]  # 确保只处理非空行
        res = json.dumps({"code": 0,
                          "msg": "success",
                          "result": sid_list})
    return res


@app.route("/download_model", methods=['POST'])
def download_model():
    info = request.get_json()
    sid_list = info['speaker_list']
    res = []
    for sid in sid_list:
        try:
            utils_audio.download_from_qiniu(sid+"_sovits", R.get_sovits_fp(sid))
            utils_audio.download_from_qiniu(sid+"_gpt", R.get_gpt_fp(sid))
            res.append({"model_name": sid, "download_success": True})
        except Exception as e:
            logger.error(f"error when download '{sid}': {repr(e.message)}")
            res.append({"model_name": sid, "download_success": False})

    res = json.dumps({"code": 0,
                      "msg": "",
                      "result": res})
    return res


# 检查云上是否已经有有这个模型可以下载
@app.route("/is_model_oss_available", methods=['POST'])
def is_model_oss_available():
    """
    模型训练是否完成
    http params:
    - speaker_list: list[str]
    """
    info = request.get_json()
    speaker_list = info['speaker_list']
    if "speaker_list" not in info:
        return json.dumps({"code": 1, "msg": "speaker_list is required", "result": []})
    m_status = []
    for sid in speaker_list:
        cond1 = utils_audio.check_on_qiniu(R.get_sovits_osskey(sid))
        cond2 = utils_audio.check_on_qiniu(R.get_gpt_osskey(sid))
        # cond1 = os.path.exists(R.get_sovits_fp(sid))
        # cond2 = os.path.exists(R.get_gpt_fp(sid))
        m_status.append({"model_name": sid, "is_available": cond1 and cond2})
    return json.dumps({"code": 0, "msg": "", "result": m_status})


# 提供本地已经下载过的所有模型
@app.route("/get_all_exist_model", methods=['POST'])
def get_all_exist_model():
    all_sid = {*os.listdir(C.GPT_DIR), *os.listdir(C.SOVITS_DIR)}
    valid_sid = [sid for sid in all_sid
                 if os.path.exists(R.get_gpt_fp(sid)) and os.path.exists(R.get_sovits_fp(sid))]
    return json.dumps({"code": 0, "msg": "", "result": valid_sid})


# 删除本地的模型
def rm_local_model():
    info = request.get_json()
    speaker_list = info['speaker_list']
    if "speaker_list" not in info:
        return json.dumps({"code": 1, "msg": "speaker_list is required", "result": []})
    m_status = []
    for sid in speaker_list:
        try:
            shutil.rmtree(os.path.join(C.GPT_DIR, sid))
            shutil.rmtree(os.path.join(C.SOVITS_DIR, sid))
            m_status.append({"model_name": sid, "is_removed": True})
        except Exception as e:
            logger.error(f"Error when rm_local_model on '{sid}', the target_dirs: '{os.path.join(C.GPT_DIR, sid)}' '{os.path.join(C.SOVITS_DIR, sid)}'")
            m_status.append({"model_name": sid, "is_removed": False})
    return json.dumps({"code": 0, "msg": "", "result": m_status})


# 查看本地文件占用大小
def get_local_file_storage():
    status, output = getstatusoutput("du -hs autodl-tmp/* | sort -hr")
    assert status == 0
    res = dict([i.split("\t")[::-1] for i in output.split("\n")])
    return json.dumps({"code": 0, "msg": "", "result": res})


def config_log():
     # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

    # 创建按天分隔的文件处理器
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
    return logger


@app.route("/model_status", methods=['POST'])
def model_status():
    # 检查本服务加载了那些speaker模型
    res = []
    for k, v in M_dict.items():
        ready_num = len([e for e in v['load_events'] if e.is_set()])
        res.append({"model_name": k, "model_num": ready_num})

    logger.debug(str(res))
    res = json.dumps({"code": 0,
                      "msg": "",
                      "result": res})
    return res


def get_model_status():
    """
    检查当前的模型状态
    """
    res = []
    for k, v in M_dict.items():
        ready_num = len([e for e in v['load_events'] if e.is_set()])
        res.append({"model_name": k, "model_num": ready_num})
    return res


# 模块级别初始化全局变量
last_status = None
connection = None
channel = None
last_sent_time = 0  # 用于记录上次发送时间
SEND_INTERVAL = 2  # 发送间隔，单位为秒


def check_and_notify():
    """
    定时检查模型状态，并将变化通知到 RabbitMQ
    """
    global last_status,last_sent_time  # 声明 last_status 是全局变量
    global connection, channel  # 如果 connection 和 channel 也需要修改，声明它们为全局变量
    # 获取当前状态
    current_status = get_model_status()

    # 比较当前状态与上一次的状态
    if current_status != last_status or (time.time() - last_sent_time > SEND_INTERVAL):

        try:
            # 检查 RabbitMQ 连接
            if channel is None or channel.is_closed:
                connection, channel = connect_to_rabbitmq()

            res = json.dumps({
                "endpointType": endpoint_type,
                "endpointUrl": service_url,
                "modelList": current_status
            })
            # 发送消息到交换机
            channel.basic_publish(
                exchange="exchange_aiservice_model_sync",
                routing_key="",  # 指定路由键
                body=res,
                properties=PROPERTIES
            )

            # 更新状态
            last_status = current_status
            last_sent_time = time.time()  # 更新上次发送时间

        except Exception as e:
            logger.error(f"Failed to send message to RabbitMQ: {e}", exc_info=True)


def schedule_tasks():
    """
    定时任务调度
    """
    schedule.every(1).seconds.do(check_and_notify)  # 每 1 秒检查一次状态

    while True:
        schedule.run_pending()
        time.sleep(1)


def cleanup():
    print("Cleaning up resources before exiting...")
    # 给队列发送结束信号
    for k, v in M_dict.items():
        p_list = v['process_list']
        for _ in range(len(p_list)):
            v['q_inp'].put("STOP")
        for p in p_list:
            p.join()

        
if __name__ == '__main__':
    logger=config_log()
    mp.set_start_method("spawn")
    logger.info("Preparing")
    os.makedirs("voice_samples", exist_ok=True)
    os.makedirs("gpt_dir", exist_ok=True)
    os.makedirs("sovits_dir", exist_ok=True)
    M_dict = {}
    # M_dict = {
    #     # 示例数据
    #     "model1": {
    #         "load_events": [mp.Event() for _ in range(5)],
    #         "process_list": [],
    #         "q_inp": mp.Queue()
    #     },
    #     "model2": {
    #         "load_events": [mp.Event() for _ in range(3)],
    #         "process_list": [],
    #         "q_inp": mp.Queue()
    #     }
    # }

    # 检查参数数量
    if len(sys.argv) != 3:
        logger.error("Error: Incorrect number of arguments.")
        logger.error("Usage: python script.py <endpoint_type> <service_url>")
        sys.exit(1)

    # 获取参数
    endpoint_type = int(sys.argv[1])
    service_url = sys.argv[2]
    # 打印参数（用于调试）
    logger.info(f"Endpoint Type: {endpoint_type}")
    logger.info(f"Service URL: {service_url}")
    
    # 启动定时任务调度线程
    logger.info("Starting scheduled tasks...")
    task_thread = threading.Thread(target=schedule_tasks)
    task_thread.daemon = True  # 设置为守护线程，主进程退出时自动结束
    task_thread.start()
    # 启动 Flask 服务
    logger.info("Starting Flask server...")
    
    app.run(host="0.0.0.0", port=8002)


