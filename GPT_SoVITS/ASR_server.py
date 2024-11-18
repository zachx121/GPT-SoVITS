import logging
import pickle
import queue
import multiprocessing as mp
import numpy as np

#mp.set_start_method("fork")
#mp.set_start_method("forkserver")
logging.basicConfig(format='[%(asctime)s-%(process)d-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
import warnings
import numba
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

import random
import whisper.tokenizer
import utils_audio
from ASR_Speech2Text import Speech2Text
#from text2speech import Text2Speech
# from sounda_voice.text2speech import Text2Speech

from flask import Flask, render_template, request, current_app
from flask_socketio import SocketIO, emit,  join_room

from queue import Empty
import time
import audioop
import torch
import sys
import threading
import base64
import json
import os
sys.path.append("./")
from datetime import datetime
import wave

# [Model prepared]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(">>> [DEVICE]: %s" % DEVICE)

DEBUG = False
PORT = 6006
PROCESS_NUM = 2
SST_MODEL_DIR = "/root/autodl-fs/whisper"
WHISPER_MODEL = "large-v2"  # large-v2, tiny, medium
OUTPUT_DIR = "./output"
VOICE_SAMPLE_DIR = "./voice_sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
logging.info(">>> [PORT]: %s" % PORT)
logging.info(">>> [SST_MODEL_DIR]: %s" % SST_MODEL_DIR)


NAME_SPACE = '/MY_SPACE'
SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE*SAMPLE_WIDTH*CHANNELS
RMS_LAST_TIME = 0.5  # 取音频的最后多久计算RMS
RMS_HOLDER = 500  # 最后一段音频小于多少音量时，视为结束，清空buffer （有子进程轮询兜底追空音频所以不担心无法结束）
NS_PROB_HOLDER = 0.5  # 超过多少的概率，认为是非说话声音，返回空串
AUDIO_RECORD = {}


# 再启动一个进程每秒轮询SID_INFO，如果发现存在非空buffer且超过一秒没接收到新的buffer，就往queue里追加一个空音频让他触发清空
def process_loop(q_inp, sid_info, lock):
    hasAppendTag = {}
    while True:
        time.sleep(0.3)
        with lock:
            # todo sid_info里不存在的key（客户端断开连接）也要从hasAppendTag里剔除
            # 下面这个做法会报错 RuntimeError: dictionary changed size during iteration
            # cur_keys = hasAppendTag.keys()
            # target_keys = sid_info.keys()
            # for k in cur_keys:
            #     if k not in target_keys:
            #         hasAppendTag.pop(k)

            for sid, info in sid_info.items():
                ts = int(time.time()*1000)
                # logging.debug(f"轮询的进程正在检查sid:{sid}, last_act:{info['last_active_time']}")
                # lat时间不能是-1（-1表示客户端刚连接的时间）
                if info["last_active_time"] != -1 and ts - info["last_active_time"] >= 1000 and len(info["buffer"]) > 1:
                    # 避免连续对同一个sid连续追加
                    # lb = len(info["buffer"])
                    # if np.frombuffer(info["buffer"][lb - int(BYTES_PER_SEC*0.8):lb], dtype=np.int16).sum() == 0:
                    #     continue
                    if hasAppendTag.get(sid, False):
                        continue
                    logging.info(f"给sid:{sid} 追加一秒钟的空音频 (ts:{ts} lat:{info['last_active_time']} txt:{info['text']})")
                    # 写入一个1秒长度的空音频给buffer
                    t_str = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
                    buffer = np.zeros(SAMPLE_RATE*1, dtype=np.int16).tobytes()
                    pad_data = {"audio": buffer,
                                "channels": CHANNELS,
                                "sample_rate": SAMPLE_RATE,
                                "sample_width": SAMPLE_WIDTH,  # 注意服务端只会对位宽2的音频做补零
                                "language": "zh",
                                "ts": int(time.time() * 1000)
                                }
                    q_inp.put((pad_data, t_str, sid))
                    hasAppendTag[sid] = True
                else:
                    hasAppendTag[sid] = False


# 子进程用whisper处理语音转文本，并将结果写入rsp队列
def process_queue_speech2text(q_input, q_output, sid_info, lock, _pid_name, model_type=WHISPER_MODEL):
    # logging.info(_PID_NAME[os.getpid()]+"process_queue_speech2text start.")
    logging.info("process_queue_speech2text start.")  # 这个时候还没while阻塞，主进程还没执行到更新_PID_NAME字典拿不到NAME
    logging.debug(">>> Construct&Init Model (device is '%s' type is '%s')" % (DEVICE, model_type))
    # 注意这里如果提前写死了用"WHISPER_MODEL"这个全局变量，那后面__main__里修改了"WHISPER_MODEL"也没用，因为函数已经提前创建好了
    model = Speech2Text(model_type=model_type, download_root=SST_MODEL_DIR, device=DEVICE)
    model.init()
    logging.debug(">>> Construct&Init Model done.")
    logging.info("process_queue_speech2text ready.")

    while True:
        # 从队列中获取数据
        data, t_str, sid = q_input.get()
        # ts_data = data['ts']
        lang = data.get("language", None)
        if lang is None:
            logging.warning("请求未提供 'language' 参数，效果可能变差 (data.keys: %s)" % data.keys())
        elif lang not in whisper.tokenizer.LANGUAGES:
            logging.warning("'language' 参数指定了不支持的语言%s" % lang)
            lang = None
        else:
            pass
        return_details = DEBUG or data.get("return_details", "0") == "1"

        if data is None:
            logging.info(_pid_name[os.getpid()] + "data is None, break now. %s" % t_str)
            break

        t_begin = time.time()
        logging.debug(_pid_name[os.getpid()] + "Process of sid-%s-%s start." % (sid, t_str))

        eos_tag = False
        # 处理数据，这里每个sid只能交由同一个子进程处理，避免如下情况
        # 00:01 进程A在锁内追加buffer，然后即将要在锁外执行transcribe_buffer
        # 00:02 进程B在锁内判定音频结束然后清空了buffer，
        with lock:
            if sid in sid_info:
                info = sid_info[sid]
                info["buffer"] += data['audio']
                # info["last_active_time"] = ts_data
                # 根据最后一秒的音量判断当前是不是eos
                lb = len(info["buffer"])
                volume = audioop.rms(info["buffer"][lb - int(RMS_LAST_TIME * BYTES_PER_SEC):lb], SAMPLE_WIDTH)
                logging.debug(_pid_name[os.getpid()] + "    最后 %s 秒的音量: %s (buffer_len:+%s=%s text:'%s')" % (RMS_LAST_TIME, volume,len(data['audio']), lb, info["text"][:15]))
                # 持续超过90s的buffer也直接转录后清空
                elapse = lb//int(RMS_LAST_TIME * BYTES_PER_SEC)
                # todo.转录已经识别到句号也清空
                if volume <= RMS_HOLDER or elapse > 90:
                    logging.debug(_pid_name[os.getpid()] + f"    清空buffer.(音量过小={volume <= RMS_HOLDER}, 长度过长={elapse > 90})")
                    logging.debug(_pid_name[os.getpid()] + f"    清空前再转录一次文本")
                    buffer2clear = info["buffer"]
                    eos_tag = True
                    info["buffer"] = b""

                # 如果有处理buffer，在锁内就赶紧更新，不然释放掉别的进程拿到锁一看发现buffer不是空的就开始转录了
                sid_info.update({sid: info})
            else:
                # 兜底，万一出现请求的sid不在SID_INFO里记录着，就直接跳过这个请求后续的处理
                logging.error(f"出现了一个不在SID_INFO里记录的sid:{sid}, SID_INFO.keys:{sid_info.keys()}")
                continue
        logging.debug(_pid_name[os.getpid()] + "    with lock结束")

        # 尝试在锁外发送消息（主进程有新的客户端连接时，需要拿锁去更新SID_INFO，如果此时子进程占着锁发送无超时的curl，等待主进程返回结果，那就形成死锁了）
        if eos_tag:
            # 尝试在锁外转录 | 转录耗时太长会导致
            # p1拿到了"buffer+1s"空还在转录，还没来得及清空；
            # 此时p2拿数据是不等锁的，就直接拿到了新数据构成 "buffer+1s+newBuffer"；
            #   - 此时p2判断结尾0.5s显然不是空音频了
            # 所以要先清空"buffer+1s"被P1判断后直接清空
            text = model.transcribe_buffer(buffer2clear,
                                           sr_inp=SAMPLE_RATE,
                                           channels_inp=CHANNELS,
                                           language=lang,
                                           fp16=False,
                                           return_details=return_details,
                                           prob_holder=NS_PROB_HOLDER)
            if DEBUG:
                logging.debug("清空前把buffer存下来")
                utils_audio.save_audio_buffer(info["buffer"], SAMPLE_RATE,
                                              fp="debug_audio_%s_%s.wav" % (request.sid, info["last_active_time"]), dtype=np.int16)
            logging.debug(_pid_name[os.getpid()] + "    识别到结束，发送最终文本(预览前20字) '%s'" % text[:20])
            rsp = json.dumps({"text": text, "mid": "0", "trace_id": data.get("trace_id","")})
            # socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
            q_output.put((rsp, sid))
            _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_speech2text" % PORT)
        else:
            text = ""
            if len(info["buffer"]) > 0:
                logging.debug(_pid_name[os.getpid()] + "    buffer非空，开始转录")
                text = model.transcribe_buffer(info["buffer"],
                                               sr_inp=SAMPLE_RATE,
                                               channels_inp=CHANNELS,
                                               language=lang,
                                               fp16=False,
                                               return_details=return_details,
                                               prob_holder=NS_PROB_HOLDER)
                logging.debug(_pid_name[os.getpid()] + "    transcribed: '%s'" % text)
                rsp = json.dumps({"text": text, "mid": "1", "trace_id": data.get("trace_id","")})

                # queue.task_done()
                logging.debug(_pid_name[os.getpid()] + "    Process of sid-%s-%s finished.(elapsed %.4f)" % (sid, t_str, time.time() - t_begin))
                # socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
                q_output.put((rsp, sid))
                _ = os.system("curl -m 5 127.0.0.1:%s/exec_emit_speech2text" % PORT)
                logging.debug(_pid_name[os.getpid()] + "size of data_queue: %s" % q_input.qsize())
            else:
                logging.debug(_pid_name[os.getpid()] + "    buffer为空")

        with lock:
            if sid in sid_info:
                info = sid_info[sid]
                info["text"] = text
                sid_info.update({sid: info})


# Flask Service init
def create_app(sid_info, lock):
    import eventlet
    eventlet.monkey_patch()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret_key'
    socketio = SocketIO()
    socketio.init_app(app, cors_allowed_origins='*', async_mode='eventlet')
    # socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')
    # socketio.init_app(app, cors_allowed_origins='*')

    # 以「127.0.0.1:8080/debug」这个页面的访问，来触发一次服务端对客户端的socket消息发送
    # curl 127.0.0.1:8080/debug
    @app.route("/debug")
    def debug_func():
        return "message send!\n"

    # 针对NAME_SPACE域里的socket连接的「connect」事件
    @socketio.on("connect", namespace=NAME_SPACE)
    def connect_msg():
        logging.info(f"client connected. {request.sid}")
        ts = time.time()
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        info = {
            "connect_time": ts,
            "connect_time_str": ts_str,
            "last_active_time": -1,
            "buffer": b"",
            "text": ""
        }
        # app = current_app._get_current_object()
        # thread = socketio.start_background_task(target=process_queue_text2speech, app=app)
        with lock:
            sid_info.update({request.sid: info})

    # 针对NAME_SPACE域里的socket连接的「disconnect」事件
    @socketio.on("disconnect", namespace=NAME_SPACE)
    def disconnect_msg():
        logging.info(f"client disconnected. {request.sid}")
        with lock:
            sid_info.pop(request.sid)

    @socketio.on("speech2text", namespace=NAME_SPACE)
    def speech2text(data):
        data = json.loads(data)
        data["audio"] = base64.b64decode(data["audio"])
        logging.info("收到的buffer长度: %s" % len(data["audio"]))
        # 如果请求里提供了必要信息，可以尝试末尾补零
        # - 正常情况下时长、采样率等数据示例 elapse:1.0, chunk_size: 22050, buffer_len: 44100, buffer2arr_shape: (22050,)
        # - 此处要求音频位宽必须是2（后续whisper处理的时候写死了buffer转arr用的int16即2个字节，约定好的暂时不改了）
        sr = data.get('sample_rate', None)
        elapse = data.get('elapse', None)
        sw = data.get('sample_width', 2)
        if all([i is not None for i in [sr, elapse, sw]]) and sw == 2:
            chunk_size = elapse * sr
            if len(data["audio"])//sw < chunk_size:
                logging.debug("发现长度不够的buffer，末尾补零")
                pad_zeros = np.zeros(int(chunk_size-len(data["audio"])//sw), dtype=np.int16)
                pad_res = np.hstack([np.frombuffer(data["audio"], dtype=np.int16), pad_zeros])
                data["audio"] = pad_res.tobytes()
                logging.info("补零后的buffer长度: %s" % len(data["audio"]))

        sid = request.sid
        # ts = int(time.time())
        ts = int(data['ts'])
        t_str = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S")
        with lock:
            if sid in sid_info:
                info = sid_info[sid]
                old_lat = info["last_active_time"]
                info["last_active_time"] = int(data['ts'])
                new_lat = info["last_active_time"]
                sid_info.update({sid: info})
                logging.debug(f"{NAME_SPACE}/speech2text received an input.(at {ts} {t_str})")
                logging.debug(f"  and update 'last_active_time': '{old_lat}'->'{new_lat}'")
                # 如果发现客户端两次发送时间超过了一秒且当前buffer也没被清空，就插入一秒空音频
                # - 子进程之前sleep(1)改成sleep(0.2)加快轮询频率就解决了，不用主进程处理
                # if old_lat != -1 and new_lat - old_lat > 1000 and len(info["buffer"]) > 1:
                #     logging.info(" 触发主进程在事件speech2text里的追加一秒空音频")
                #     t_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
                #     buffer = np.zeros(SAMPLE_RATE * 1, dtype=np.int16).tobytes()
                #     pad_data = {"audio": buffer,
                #                 "channels": CHANNELS,
                #                 "sample_rate": SAMPLE_RATE,
                #                 "sample_width": SAMPLE_WIDTH,  # 注意服务端只会对位宽2的音频做补零
                #                 "language": "zh",
                #                 "ts": int(time.time() * 1000)
                #                 }
                #     Q_speech2text.put((pad_data, t_str, sid))
            # 将数据添加到队列中(数据放完再释放锁，因为现在有另一个进程也会往里面放数据）
            Q_speech2text.put((data, t_str, sid))
            logging.debug("size(estimate) of Q_speech2text: %s" % Q_speech2text.qsize())

    @app.route("/exec_emit_speech2text")
    def exec_emit_speech2text():
        # logging.debug("触发了向客户端发送消息（持续从队列中加载直到没有可加载的）")
        while True:
            try:
                rsp, sid = Q_speech2text_rsp.get_nowait()
                # logging.debug("    队列拿到结果 —— rsp:'%s...'" % str(rsp[:15]))
                socketio.emit("speech2text_rsp", rsp, to=sid, namespace=NAME_SPACE)
            except Empty:
                # logging.debug("    队列为空，退出")
                break
        return ""

    return app, socketio


# python server_vits_mp_speech2text.py 3 medium
if __name__ == '__main__':
    PROCESS_NUM = int(sys.argv[1]) if len(sys.argv) >= 2 else PROCESS_NUM
    WHISPER_MODEL = sys.argv[2] if len(sys.argv) >= 3 else WHISPER_MODEL
    logging.info(">>> 并行进程数量: %s" % PROCESS_NUM)
    logging.info(">>> 使用whisper模型: %s" % WHISPER_MODEL)

    mp.set_start_method("forkserver")
    manager = mp.Manager()
    Q_speech2text = manager.Queue()
    Q_speech2text_rsp = manager.Queue()
    SID_INFO = manager.dict()
    LOCK = mp.Lock()
    _PID_NAME = manager.dict()
    _PID_NAME[os.getpid()] = "主进程"
    # processes = []
    for idx in range(PROCESS_NUM):
        p1 = mp.Process(target=process_queue_speech2text,
                        args=(Q_speech2text, Q_speech2text_rsp, SID_INFO, LOCK, _PID_NAME, WHISPER_MODEL))
        p1.start()
        # processes.append(p1)
        _PID_NAME.update({p1.pid: "子进程%s" % idx})
        logging.info("    speech2text子进程启动 (%s)" % p1.pid)

    p1 = mp.Process(target=process_loop, args=(Q_speech2text, SID_INFO, LOCK))
    p1.start()
    logging.info("    轮询检查并追加空buffer的子进程启动 (%s)" % p1.pid)

    logging.info(">>> MultiProcess ready. all pid as follow:")
    logging.info(_PID_NAME)

    app, socketio = create_app(SID_INFO, LOCK)
    logging.info("\n"*2+">>> Socket App run..."+"\n"*2)
    socketio.run(app, host='0.0.0.0', port=PORT, debug=False)  # 阻塞
    # gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 server_sounda_mp:app
