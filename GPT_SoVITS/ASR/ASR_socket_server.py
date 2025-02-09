from flask import Flask
from flask_socketio import SocketIO, emit
import threading
import numpy as np
from funasr import AutoModel

# 初始化 Flask 应用和 SocketIO
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 6006
app = Flask(__name__)
socketio = SocketIO(app)
# 同一时间只允许一个客户端连接socket
is_connected, is_connected_lock = False, threading.Lock()

# 初始化模型和流式参数
# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
model = AutoModel(model="/root/GPT-SoVITS/GPT_SoVITS/ASR/model/paraformer-zh-streaming", model_revision="v2.0.4")
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
cache, cache_lock = {}, threading.Lock()

@socketio.on('connect', namespace="/ASR")
def handle_connect():
    global cache, is_connected
    with is_connected_lock:
        if is_connected:
            emit('error', 'Another client is already connected.')
            return False
        is_connected = True
    # 连接开始，清空缓存
    print(f"on connect, clean 'cache'")
    with cache_lock:
        cache = {}

@socketio.on('disconnect', namespace="/ASR")
def handle_disconnect():
    global cache, is_connected
    # 连接结束，清空缓存
    with cache_lock:
        cache = {}
    with is_connected_lock:
        is_connected = False

@socketio.on('audio_data', namespace="/ASR")
def handle_audio_data(audio_buffer):
    global model, cache
    print("server audio_data received")
    try:
        # 将字节流转换为音频数组
        audio_arr = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"audio_arr: shape {audio_arr.shape}")
        # 进行语音识别
        res = model.generate(input=audio_arr, cache=cache,
                             is_final=False, chunk_size=chunk_size,
                             encoder_chunk_look_back=encoder_chunk_look_back,
                             decoder_chunk_look_back=decoder_chunk_look_back,
                             disable_pbar=True)

        # 返回结果
        print(f"res: {res}")
        emit('result', str(res[0]['text']))
    except Exception as e:
        print(f"error: {str(e)}")
        emit('error', str(e))


if __name__ == "__main__":
    socketio.run(app, host=SERVER_HOST, port=SERVER_PORT, debug=True)

