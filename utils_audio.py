import scipy
import numpy as np
import audioop
from urllib.parse import unquote
from subprocess import getstatusoutput
import requests
import logging
import multiprocessing as mp

SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

def play_audio_buffer(audio_buffer, sr, channels=1):
    # pyaudio在有些环境出现兼容问题，就不在外部统一import了
    import pyaudio
    p = pyaudio.PyAudio()
    # 打开一个音频流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    output=True)
    # 播放音频
    stream.write(audio_buffer)
    # 结束后关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

play_audio = play_audio_buffer

def save_audio_buffer(audio_buffer, sr, fp, dtype=np.float32):
    scipy.io.wavfile.write(fp, sr, np.frombuffer(audio_buffer, dtype=dtype))

def save_audio(audio, sr, fp):
    scipy.io.wavfile.write(fp, sr, audio)
    
# 计算音量，默认每0.5s一个计算gap
def cal_rms(inp_buffer, delta=0.5, sr=SAMPLE_RATE, sw=SAMPLE_WIDTH, c=CHANNELS):
    bps = sr*sw*c
    total_time = len(inp_buffer) / bps
    volume = []
    ts = []
    for i in range(0, int(total_time / delta)):
        s = int(i * delta * bps)
        e = int((i + 1) * delta * bps)
        y = audioop.rms(inp_buffer[s:e], sw)
        volume.append(y)
        ts.append(i*delta)
    return volume, ts


# wave写的wav文件dtype应该是用np.int16来解析
def play_audio_buffer_with_volume(audio_buffer, sr, channels=1, dtype=np.int16):
    import pyaudio
    p = pyaudio.PyAudio()

    # 打开一个音频流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    output=True)

    # 定义音频块大小
    BYTES_PER_SEC = sr * channels * 2
    CHUNK = int(1*BYTES_PER_SEC)

    # 转换音频缓冲区到 NumPy 数组
    audio_np_array = np.frombuffer(audio_buffer, dtype=dtype)

    # 播放音频
    for i in range(0, len(audio_np_array), CHUNK):
        chunk = audio_np_array[i:i+CHUNK]
        # 将 NumPy 数组转换为字节，写入音频流
        stream.write(chunk.astype(dtype).tobytes())

        # 计算并打印音量
        rms = audioop.rms(chunk.astype(dtype).tobytes(), 2)  # Here width = 2 because we're considering int16
        print("Volume:", rms)

    # 结束后关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()


def get_latest_fp(inp_dir):
    _inp_dir = os.path.expanduser(inp_dir)
    fp = sorted(os.listdir(_inp_dir),
                key=lambda x: os.path.getmtime(os.path.join(_inp_dir, x)), reverse=True)[0]
    return fp


def download_file(url, target_dir):
    """ 下载文件并保存到指定目录 """
    filename = os.path.basename(unquote(url).split("?")[0])
    file_path = os.path.join(target_dir, filename)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded: {filename}")
        else:
            # 当状态码不是 200 时，抛出异常
            raise Exception(f"Failed to download {url}, status code {response.status_code}")
    except Exception as e:
        # 捕获异常并重新抛出
        raise Exception(f"Error downloading {url}: {str(e)}")


def download_files_in_parallel(urls, target_dir, num_workers=4):
    """ 并行下载多个文件 """
    with mp.Pool(num_workers) as pool:
        try:
            # 异步执行下载任务
            results = [pool.apply_async(download_file, (url, target_dir)) for url in urls]
            # 获取每个任务的结果
            for result in results:
                result.get()  # 获取结果，如果有异常会在这里抛出
        except Exception as e:
            # 当捕获到异常时，终止进程池并重新抛出异常
            pool.terminate()
            pool.join()
            raise Exception(f"Download failed: {str(e)}")


from qiniu import Auth, put_file, etag, BucketManager
import qiniu.config
import os
class QiniuConst:
    access_key = "izz8Pq4VzTJbD8CmM3df5BAncyqynkPgF1K4srqP"
    secret_key = "pOhSAES6tocA3PzNF2fS_bnShTLUX5TEA1-tUmJY"
    bucket_domain = "http://resource.aisounda.cn"
    bucket_public_domain = "https://public.yisounda.com"
    bucket_name = 'sounda'
    bucket_public_name = 'sounda-public'


def post2qiniu(localfile, key):
    os.path.exists(localfile)

    # >>> 上传
    q = Auth(QiniuConst.access_key, QiniuConst.secret_key)
    token = q.upload_token(QiniuConst.bucket_name, key, 3600)  # 生成上传Token，可以指定token的过期时间等
    ret, info = put_file(token, key, localfile, version='v2')
    # print(info)
    assert ret['key'] == key
    assert ret['hash'] == etag(localfile)
    # print(f"wget -O res '{private_url}'")  # >>> 下载路径
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url('%s/%s' % (QiniuConst.bucket_domain, key), expires=3600)
    return private_url


def check_on_qiniu(keys, bucket_name=QiniuConst.bucket_name):
    if isinstance(keys, str):
        keys = [keys]
    q = Auth(QiniuConst.access_key, QiniuConst.secret_key)
    # 初始化BucketManager
    bucket = BucketManager(q)
    ret, eof, info = bucket.list(bucket_name)
    return [i['key'] for i in ret['items'] if all(j in i['key'] for j in keys)]


def get_url_from_qiniu(key, domain=QiniuConst.bucket_domain):
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url('%s/%s' % (domain, key), expires=3600)
    return private_url


def download_from_qiniu(key, fp):
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url('%s/%s' % (QiniuConst.bucket_domain, key), expires=3600)
    cmd1 = f"mkdir -p '{os.path.dirname(fp)}'"
    cmd2 = f"wget --no-check-certificate -O '{fp}' '{private_url}'"
    # cmd2 = f"wget -O {fp} '{private_url}'"
    s, o = getstatusoutput(f"{cmd1} && {cmd2}")
    assert s == 0, f"download failed. output:{o}"

if __name__ == '__main__':
    print(f"""wget -O 'G2PWModel.tgz' '{get_url_from_qiniu("models/G2PWModel.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(f"""wget -O 'GSV_pretrained_models.tgz' '{get_url_from_qiniu("models/GSV_pretrained_models.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(f"""wget -O 'ChineseASR_Damo.tgz' '{get_url_from_qiniu("ChineseASR_Damo.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(f"""wget -O 'faster_whisper_large_v3.tgz' '{get_url_from_qiniu("models/faster_whisper_large_v3.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    # print(check_on_qiniu("model/clone/device/20250211/1000294265/"))
