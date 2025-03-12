import scipy
import numpy as np
import audioop
from matplotlib import pyplot as plt
from urllib.parse import unquote
from subprocess import getstatusoutput
import requests
import logging
import multiprocessing as mp
from qiniu import Auth, put_file, etag, BucketManager
import qiniu.config
import os
import librosa

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
    bps = sr * sw * c
    total_time = len(inp_buffer) / bps
    volume = []
    ts = []
    for i in range(0, int(total_time / delta)):
        s = int(i * delta * bps)
        e = int((i + 1) * delta * bps)
        y = audioop.rms(inp_buffer[s:e], sw)
        volume.append(y)
        ts.append(i * delta)
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
    CHUNK = int(1 * BYTES_PER_SEC)

    # 转换音频缓冲区到 NumPy 数组
    audio_np_array = np.frombuffer(audio_buffer, dtype=dtype)

    # 播放音频
    for i in range(0, len(audio_np_array), CHUNK):
        chunk = audio_np_array[i:i + CHUNK]
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


def batch_check_samples():
    import subprocess
    from glob import glob
    import logging
    for ASR_RES_FP in glob("/root/autodl-fs/voice_sample/*/asr/denoised.list"):
        print(f"at: {ASR_RES_FP}")
        res = subprocess.run(["wc", "-l", ASR_RES_FP], capture_output=True, text=True, encoding='utf-8')
        assert res.returncode == 0
        sample_num = int(res.stdout.strip().split(" ")[0])
        if sample_num <= 10:
            logging.error(f"样本数量异常(仅{sample_num}个), {ASR_RES_FP}")
            for i in glob(ASR_RES_FP.split("/asr/denoised.list")[0] + "/*.wav"):
                if "ref_audio" in i:
                    continue
                print(i)
                # Audio(i)


def vis_phones_and_bert(phones, bert, norm_text):
    if type(bert) != np.ndarray:
        bert = bert.cpu().numpy()
    print(f"norm_text2: '{norm_text}'")
    print(f"token-num:{len(norm_text.split(' '))} char-num:{len(norm_text)} 音素数量:{len(phones)}, bert.shape:{bert.shape}")
    fig,axs = plt.subplots(1,2, figsize=(9,3))
    _ = axs[0].set_title(f"bert min~max of {bert.shape[1]} items")
    arr = bert
    _ = axs[0].fill_between(np.arange(arr.shape[1]),
                         y1=np.min(arr, axis=0),
                         y2=np.max(arr, axis=0),
                         color='steelblue')
    _ = axs[0].fill_between(np.arange(arr.shape[1]),
                         y1=np.quantile(arr, q=0.05, axis=0),
                         y2=np.quantile(arr, q=0.95, axis=0),
                         color='red', alpha=0.3)
    _ = axs[0].plot(np.arange(arr.shape[1]), np.quantile(arr, q=0.5, axis=0), color='red', label="q50")
    _ = axs[0].plot(np.arange(arr.shape[1]), np.min(arr, axis=0), color='steelblue', label="min")
    _ = axs[0].plot(np.arange(arr.shape[1]), np.max(arr, axis=0), color='steelblue', label="max")
    _ = axs[1].set_title(f"phones_id of {len(phones)} items")
    _ = axs[1].plot(np.arange(len(phones)), phones)
    _ = axs[1].scatter(np.arange(len(phones)), phones)
    plt.show()


class NoiseCheck:
    @staticmethod
    def consecutive_true(arr, num=3):
        for i in range(0, len(arr)):
            if all(arr[i:i + num]) and len(arr[i:i + num]) == num:
                return True
        return False

    @staticmethod
    def consecutive_low_var(arr, var_hold=0.01, value_hold=1, num=3):
        for i in range(0, len(arr)):
            if len(arr[i:i + num]) == num and np.var(arr[i:i + num]) <= var_hold and all(arr[i:i + num] <= value_hold):
                return True, i, i + num
        return False, -1, -1

    @staticmethod
    def is_abnormal_pronounce(y, sr, debug=False):
        if y.shape[0] / sr <= 3.0:
            logging.info("    音频不足3.0秒，不进行峰度和能量的异常检测")
            return False
        if y.dtype not in (np.float16, np.float32):
            y = y.astype(np.float32) / 32768.0
        # hop_length: 滑动窗口要滑多少个采样的样本，比如设为frame_length的2/3时表示滑窗会移动2/3长度，即末尾1/3和下一个开头是重叠的
        winframe = int(0.5 * sr)  # 0.5秒
        frame_length = winframe
        hop_length = winframe
        consecutive_num = max(3, int((y.shape[0] / winframe)*0.25))

        # frames = librosa.util.frame(y, frame_length=winframe, hop_length=winframe//3*2)
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        logging.debug(f"{frames.shape[1]}个frame片段")

        # 计算能量
        energy = np.sum(np.abs(frames) ** 2, axis=0)
        # 频谱平坦度
        # flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length)[0][:len(energy)]
        # 峰度
        zcr = np.array([scipy.stats.kurtosis(signal) for signal in frames.T])

        energy_hold = max(1, np.quantile(energy, 0.75) * 0.5)
        cond0 = all(energy[:3] <= energy_hold) or all(energy[1:4] <= energy_hold)
        if cond0: logging.info(f"    检测到异常音频 (开头低能量)")

        cond2, i, j = NoiseCheck.consecutive_low_var(zcr, var_hold=0.01, value_hold=1, num=consecutive_num)
        zcr_cond_consecutive = np.zeros_like(zcr, dtype=bool)
        zcr_cond_consecutive[i:j] = True
        if cond2: logging.info(f"    检测到异常音频 (峰度不足1且连续无变化)")

        energy_cond = energy <= energy_hold
        # 首尾如果峰度太高，可能是静音频段突变人声，需要剔除
        s, e = 1 if zcr[0] >= 20 else 0, -1 if zcr[-1] >= 20 else len(zcr)
        zcr_qtl = np.quantile(zcr[s:e], 0.75)
        zcr_hold = max(0.5, zcr_qtl * 0.3)
        zcr_cond = zcr <= zcr_hold
        cond3 = NoiseCheck.consecutive_true(zcr_cond, num=consecutive_num)
        if cond3: logging.info(f"    检测到异常音频 (连续低峰度)")

        if debug:
            # logging.info(f"energy: {energy.tolist()}, avg:{np.mean(energy)}, hold:{np.mean(energy)*0.7}")
            # logging.info(f"flatness: {flatness.tolist()}, avg:{np.mean(flatness)}, hold:{np.mean(flatness)*0.7}")
            # logging.info(f"zcr: {zcr.tolist()}, avg:{np.mean(zcr)}, hold:{np.mean(zcr)*0.7}")

            fig, axs = plt.subplots(2, 2, figsize=(9, 4))
            fig.set_tight_layout(True)
            axs = axs.flatten()
            _ = plt.figure(figsize=(6, 2))
            for idx, f in enumerate(frames.T):
                _ = axs[0].plot(range(idx * winframe, (idx + 1) * winframe), f)
                _ = axs[0].set_title("Original Wav")
                _ = axs[1].plot(range(idx * winframe, (idx + 1) * winframe), f)
                _ = axs[1].set_title(f"Energy_{energy_hold:.2f}")
                _ = axs[1].text(idx * winframe, np.max(y) * 1.1, f"{energy[idx]:.0f}", c='red', rotation=60)
                _ = axs[2].plot(range(idx * winframe, (idx + 1) * winframe), f)
                _ = axs[2].set_title("Kurtosis_Consecutive_LowVar")

                _ = axs[3].plot(range(idx * winframe, (idx + 1) * winframe), f)
                _ = axs[3].set_title(f"Kurtosis_{zcr_hold:.2f}")
                _ = axs[3].text(idx * winframe, np.max(y) * 1.1, f"{zcr[idx]:.4f}", c='red', rotation=60)
                if energy_cond[idx]:
                    _ = axs[1].axvspan(idx * winframe, (idx + 1) * winframe, color='red', alpha=0.3)

                if zcr_cond_consecutive[idx]:
                    _ = axs[2].axvspan(idx * winframe, (idx + 1) * winframe, color='blue', alpha=0.3)

                if zcr_cond[idx]:
                    _ = axs[3].axvspan(idx * winframe, (idx + 1) * winframe, color='yellow', alpha=0.3)

            _ = plt.show()

        return cond0 or cond2 or cond3

    @staticmethod
    def standalone_test():
        check = NoiseCheck.is_abnormal_pronounce
        from IPython.display import Audio, Image
        #####
        from glob import glob
        badcase = glob("/root/autodl-fs/audio_samples/abnormal_audio/*.wav")
        logging.info(f"total: {len(badcase)}")
        for fp in badcase:
            y, sr = librosa.load(fp, sr=None)
            logging.info(f">>> at fp: {fp}")
            # 漏召回
            if not check(y, sr, debug=False):
                Audio(fp)
                check(y, sr, debug=True)

        goodcase = glob("/root/autodl-fs/voice_sample/*/ref_audio_default.wav")
        logging.info(f"total: {len(goodcase)}")
        for fp in goodcase:
            y, sr = librosa.load(fp, sr=None)
            logging.info(f">>> at fp: {fp}")
            # 误检出
            if check(y, sr, debug=False):
                Audio(fp)
                check(y, sr, debug=True)


        goodcase = glob("/root/autodl-fs/audio_samples/audio_test_lydia/*.wav")
        logging.info(f"total: {len(goodcase)}")
        for fp in goodcase:
            y, sr = librosa.load(fp, sr=None)
            logging.info(f">>> at fp: {fp}")
            # 误检出
            if check(y, sr, debug=False):
                Audio(fp)
                check(y, sr, debug=True)


        goodcase = glob("/root/autodl-fs/audio_samples/audio_test_amber/*.wav")
        logging.info(f"total: {len(goodcase)}")
        for fp in goodcase:
            y, sr = librosa.load(fp, sr=None)
            logging.info(f">>> at fp: {fp}")
            # 误检出
            if check(y, sr, debug=False):
                Audio(fp)
                check(y, sr, debug=True)


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
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url(
        '%s/%s' % (QiniuConst.bucket_domain, key), expires=3600)
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
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url('%s/%s' % (domain, key),
                                                                                          expires=3600)
    return private_url


def download_from_qiniu(key, fp):
    private_url = Auth(QiniuConst.access_key, QiniuConst.secret_key).private_download_url(
        '%s/%s' % (QiniuConst.bucket_domain, key), expires=3600)
    cmd1 = f"mkdir -p '{os.path.dirname(fp)}'"
    cmd2 = f"wget --no-check-certificate -O '{fp}' '{private_url}'"
    # cmd2 = f"wget -O {fp} '{private_url}'"
    s, o = getstatusoutput(f"{cmd1} && {cmd2}")
    assert s == 0, f"download failed. output:{o}"


if __name__ == '__main__':
    print(
        f"""wget -O 'G2PWModel.tgz' '{get_url_from_qiniu("models/G2PWModel.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(
        f"""wget -O 'GSV_pretrained_models.tgz' '{get_url_from_qiniu("models/GSV_pretrained_models.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(
        f"""wget -O 'ChineseASR_Damo.tgz' '{get_url_from_qiniu("ChineseASR_Damo.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    print(
        f"""wget -O 'faster_whisper_large_v3.tgz' '{get_url_from_qiniu("models/faster_whisper_large_v3.tgz", domain=QiniuConst.bucket_public_domain)}'""")
    # print(check_on_qiniu("model/clone/device/20250211/1000294265/"))
