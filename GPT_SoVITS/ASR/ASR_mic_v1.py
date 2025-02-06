from funasr import AutoModel
import pyaudio
import numpy as np
import time
import audioop
import multiprocessing as mp
import scipy
# 流式参数设置
chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

# # 加载模型
# model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

# 音频参数设置
FORMAT, WIDTH = pyaudio.paInt16, 2
CHANNELS = 1
RATE = 16000
BYTES_PER_SEC = RATE * WIDTH * CHANNELS  # 每秒字节流大小
# CHUNK = chunk_size[1] * 960  # 600ms  # 必须是9600即16khz下的0.6秒，不然推理不出来？
# 调整CHUNK大小为0.1秒的数据量
SAMPLE_SEC = 0.1  # 音频流每隔多久触发一次回调
CHUNK = int(RATE * SAMPLE_SEC)

SILENCE_RMS = 500  # 低于500的认为是空白音
SILENCE_DURATION = 0.5  # 连续0.5秒空白音就认为是停顿，置is_final=True



def process_audio(queue, event):
    # 加载模型
    model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")
    event.set()
    cache = {}
    input_buffer = []
    start_accum = False
    while True:
        try:
            # 从队列中获取数据
            data, cur_duration, cur_rms = queue.get()
            if not start_accum and cur_rms >= SILENCE_RMS:
                # 有声音时开始追加音频片段到buffer中
                start_accum = True

            if start_accum:
                # 将二进制数据转换为 numpy 数组
                input_buffer.append(data)
            accum_duration = round(len(input_buffer) * SAMPLE_SEC, 2)

            print(f"""expect-duration: {CHUNK / RATE:.2f} read-duration: {cur_duration} read-rms {cur_rms} accum-duration:{accum_duration}""")

            if accum_duration >= 0.6:
                # 检查最后三个片段是否静音片段
                if audioop.rms(b"".join(input_buffer[-3:]), WIDTH) < SILENCE_RMS:
                    is_final = True
                    start_accum = False  # 出现静音片段时，重新开始accum
                else:
                    is_final = False
                # 六个字节流拼接后转成音频数组
                speech_chunk = np.frombuffer(b"".join(input_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                scipy.io.wavfile.write(f"tmp_06s_{time.time():.02f}.wav", RATE, speech_chunk)
                # 清空buffer
                input_buffer = []
                continue
                # 进行实时语音识别
                res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
                                     encoder_chunk_look_back=encoder_chunk_look_back,
                                     decoder_chunk_look_back=decoder_chunk_look_back,
                                     disable_pbar=True)
                print(f"""text: {res[0]['text']}""")
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    # 创建队列
    queue = mp.Queue()
    event = mp.Event()
    # 创建子进程
    p_process = mp.Process(target=process_audio, args=(queue, event))
    p_process.start()

    while not event.is_set():
        pass
    print("[Done]子进程模型加载完成")
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开麦克风流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* 开始录音")

    try:
        while True:
            sta = time.time()
            # 从麦克风读取音频数据
            data = stream.read(CHUNK)
            cur_duration = round(len(data) / BYTES_PER_SEC, 2)
            cur_rms = audioop.rms(data, WIDTH)
            # 将数据放入队列
            queue.put((data, cur_duration, cur_rms))
    except KeyboardInterrupt:
        print("* 录音停止")
    finally:
        # 停止流并关闭 PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        # 终止子进程
        p_process.terminate()
        p_process.join()