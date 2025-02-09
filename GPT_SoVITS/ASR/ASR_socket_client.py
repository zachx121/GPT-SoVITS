import pyaudio
import time
import socketio


# 服务端配置
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8888
sio = socketio.Client()
# 连接到服务器
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8888
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}'
SERVER_URL = "https://u212392-9062-d5043001.bjb1.seetacloud.com:8443"



# 音频参数设置
FORMAT, WIDTH = pyaudio.paInt16, 2
CHANNELS = 1
RATE = 16000
BYTES_PER_SEC = RATE * WIDTH * CHANNELS  # 每秒字节流大小
RECORD_SECONDS = 0.6
CHUNK = 1024  # 每次从麦克风读取的数据流
# 计算 0.6 秒音频数据对应的字节数
BYTES_0_6_SECONDS = int(BYTES_PER_SEC * RECORD_SECONDS)


# 定义处理服务器返回结果的事件
@sio.on('result', namespace='/ASR')
def handle_result(result):
    print(f"Received result: {result}")


if __name__ == "__main__":
    # 连接到服务器
    sio.connect(SERVER_URL, namespaces=['/ASR'])
    print(f"Connected to server {SERVER_HOST}:{SERVER_PORT}")
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print(f"Mic ready.")


    try:
        accumulated_data = b''
        while True:
            # 读取音频数据块
            data = stream.read(CHUNK)
            accumulated_data += data

            # 当积累到 0.6 秒的音频数据时
            if len(accumulated_data) >= BYTES_0_6_SECONDS:
                # 截取 0.6 秒的数据
                audio_data = accumulated_data[:BYTES_0_6_SECONDS]

                # 发送音频数据到服务端
                # client_socket.sendall(audio_data)
                sio.emit('audio_data', audio_data, namespace='/ASR')

                print(f"Sent {len(audio_data)} bytes of audio data to server.")

                # 处理剩余未发送的数据
                accumulated_data = accumulated_data[BYTES_0_6_SECONDS:]

    except KeyboardInterrupt:
        print("Interrupted by user. Stopping...")
    finally:
        # 关闭音频流和 PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 关闭 Socket 连接
        # client_socket.close()
