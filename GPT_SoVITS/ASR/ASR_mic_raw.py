from funasr import AutoModel
import pyaudio
import numpy as np

# 流式参数设置
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

# 加载模型
model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

# 音频参数设置
CHUNK = chunk_size[1] * 960  # 600ms
FORMAT,WIDTH = pyaudio.paInt16,2
CHANNELS = 1
RATE = 16000
BYTES_PER_SEC = RATE * WIDTH * CHANNELS  # 每秒字节流大小

SAMPLE_SEC = 0.05  # 音频流每隔多久触发一次回调

SILENCE_RMS = 500  # 低于500的认为是空白音
SILENCE_DURATION = 0.5  # 连续0.5秒空白音就认为是停顿，置is_final=True


# 初始化 PyAudio
p = pyaudio.PyAudio()

# 打开麦克风流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* 开始录音")

cache = {}
try:
    while True:
        # 从麦克风读取音频数据
        data = stream.read(CHUNK)
        # 将二进制数据转换为 numpy 数组
        speech_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # 模拟是否为最后一个块，这里假设一直不是最后一个块，除非手动停止
        is_final = False

        # 进行实时语音识别
        res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
                             encoder_chunk_look_back=encoder_chunk_look_back,
                             decoder_chunk_look_back=decoder_chunk_look_back,
                             disable_pbar=True)
        print(res)

except KeyboardInterrupt:
    print("* 录音停止")

finally:
    # 停止流并关闭 PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()