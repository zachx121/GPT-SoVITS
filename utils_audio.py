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


def resample_audio(audio_arr_int16, original_rate=32000, target_rate=16000):
    # 计算重采样后的长度
    resampled_length = int(len(audio_arr_int16) * (target_rate / original_rate))
    # 进行重采样
    resampled_audio = scipy.signal.resample(audio_arr_int16, resampled_length)
    # 将重采样后的音频数据转换回 int16 类型
    resampled_audio_int16 = np.int16(resampled_audio)
    return resampled_audio_int16


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

    @staticmethod
    def detect_constant_std_segments(audio_waveform, sr, frame_length=0.1, hop_length=0.05, diff_threshold=0.01,
                                     min_constant_frames=10, max_cumulative_std_deviation=0.02, plot_results=False):
        """
        检测音频波形中帧间标准差保持不变（或极小变化）的片段，并可选择绘制结果。
        同时，增加逻辑限制连续不变段内标准差的总变动范围。

        # frame_length 太小的话，比如0.01相当于检测第N个和第N+1个0.01s之间的方差变化，这个太陡峭了，用0.1平滑一点
        # min_constant_frames 持续10帧，也就是10*frame_length=10*0.1=1秒

        参数:
        audio_waveform (np.array): 输入的音频波形数据（一维NumPy数组）。
        sr (int): 音频的采样率。
        frame_length (float): 每一帧的长度（秒）。
        hop_length (float): 帧之间的跳跃长度（秒），通常小于frame_length以实现重叠。
        diff_threshold (float): 连续帧标准差之间允许的最大绝对差值，低于此值视为“不变”。
                                这个值通常应该非常小，接近浮点数的精度限制，例如 1e-5 到 1e-7。
        min_constant_frames (int): 帧标准差保持不变的最短持续帧数。
                                    例如，如果每秒100帧，50帧代表0.5秒。
        max_cumulative_std_deviation (float): 允许的连续不变段内，标准差的最大累积总变动。
                                              即，(max(segment_std) - min(segment_std)) <= max_cumulative_std_deviation。
        plot_results (bool): 是否绘制结果图。

        返回:
        list: 一个列表，每个元素是一个元组 (start_time, end_time)，表示检测到的帧标准差不变的音频段。
        """

        frame_size = int(frame_length * sr)
        hop_size = int(hop_length * sr)

        frames = librosa.util.frame(audio_waveform, frame_length=frame_size, hop_length=hop_size, axis=0)

        print(f"Frames shape after librosa.util.frame: {frames.shape}")

        # 计算每一帧内部的标准差 (形状: (num_frames,))
        frame_stds = np.std(frames, axis=1)

        # 计算相邻帧标准差的绝对差值
        std_diffs = np.abs(np.diff(frame_stds))

        # 判断哪些帧间变化是“不变”的（小于 diff_threshold）
        is_constant_diff = std_diffs < diff_threshold

        abnormal_segments = []
        current_constant_run_start_idx = -1  # 在 is_constant_diff 数组中的起始索引

        # Track min/max std within the current potential constant segment
        current_segment_min_std = float('inf')
        current_segment_max_std = float('-inf')

        # 遍历 is_constant_diff 数组来寻找连续的 True 序列
        for i in range(len(is_constant_diff)):
            # 更新当前帧的 STD
            current_frame_std = frame_stds[i]  # This is frame_stds[i]
            next_frame_std = frame_stds[i + 1]  # This is frame_stds[i+1] if i < len(frame_stds) - 1

            if is_constant_diff[i]:
                if current_constant_run_start_idx == -1:
                    # 新的连续不变序列开始
                    current_constant_run_start_idx = i
                    # 初始化该序列的 min/max STD
                    current_segment_min_std = min(current_frame_std, next_frame_std)
                    current_segment_max_std = max(current_frame_std, next_frame_std)
                else:
                    # 扩展现有序列
                    current_segment_min_std = min(current_segment_min_std, next_frame_std)
                    current_segment_max_std = max(current_segment_max_std, next_frame_std)

                # 实时检查当前序列的总变动是否超出限制
                if (current_segment_max_std - current_segment_min_std) > max_cumulative_std_deviation:
                    # 即使帧间变化小，但总变动太大，中断当前序列
                    # 检查中断前是否已形成有效片段
                    if current_constant_run_start_idx != -1:  # Ensure there was a segment in progress
                        # 这里的 i 是导致超出的那一个 std_diffs 的索引
                        # 所以有效的恒定帧序列是到 i 之前的那个点
                        # 连续不变的 diffs 数量： i - current_constant_run_start_idx
                        # 这意味着有 (i - current_constant_run_start_idx) + 1 个帧的 STD 是恒定的
                        num_constant_frames_in_segment = (i - current_constant_run_start_idx) + 1

                        if num_constant_frames_in_segment >= min_constant_frames:
                            # 记录当前有效的片段
                            start_time = current_constant_run_start_idx * hop_length
                            end_time = (
                                                   current_constant_run_start_idx + num_constant_frames_in_segment - 1) * hop_length
                            abnormal_segments.append((start_time, end_time))
                    current_constant_run_start_idx = -1  # 重置，等待下一个连续序列
                    # 重置 min/max STD，因为我们开始了新的潜在序列
                    current_segment_min_std = float('inf')
                    current_segment_max_std = float('-inf')
            else:  # is_constant_diff[i] 为 False，连续性中断
                if current_constant_run_start_idx != -1:
                    num_constant_frames_in_segment = (i - current_constant_run_start_idx) + 1

                    # 再次检查总变动，确保在序列结束时也满足条件
                    if (current_segment_max_std - current_segment_min_std) <= max_cumulative_std_deviation and \
                            num_constant_frames_in_segment >= min_constant_frames:
                        start_time = current_constant_run_start_idx * hop_length
                        end_time = (current_constant_run_start_idx + num_constant_frames_in_segment - 1) * hop_length
                        abnormal_segments.append((start_time, end_time))
                    current_constant_run_start_idx = -1  # 重置，等待下一个连续序列
                    # 重置 min/max STD
                    current_segment_min_std = float('inf')
                    current_segment_max_std = float('-inf')

        # 处理最后一个可能持续到数组末尾的连续不变序列
        if current_constant_run_start_idx != -1:
            num_constant_frames_in_segment = (len(is_constant_diff) - current_constant_run_start_idx) + 1

            # 最后的检查
            if (current_segment_max_std - current_segment_min_std) <= max_cumulative_std_deviation and \
                    num_constant_frames_in_segment >= min_constant_frames:
                start_time = current_constant_run_start_idx * hop_length
                end_time = (current_constant_run_start_idx + num_constant_frames_in_segment - 1) * hop_length
                abnormal_segments.append((start_time, end_time))

        # --- 绘制结果 ---
        if plot_results:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

            # --- 子图1: 音频波形图 ---
            time = np.linspace(0, len(audio_waveform) / sr, len(audio_waveform))
            ax1.plot(time, audio_waveform, color='blue', alpha=0.7, label='Audio Waveform')

            for start_t, end_t in abnormal_segments:
                ax1.axvspan(start_t, end_t, color='red', alpha=0.3)

            ax1.set_title('Audio Waveform with Detected Constant STD Segments')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()

            # --- 子图2: 帧标准差图 ---
            frame_time = np.arange(len(frame_stds)) * hop_length

            ax2.plot(frame_time, frame_stds, color='green', label='Frame Standard Deviation')
            ax2.scatter(frame_time, frame_stds, color='green')

            # 可选：绘制 diff_threshold
            # diff_time = np.arange(len(std_diffs)) * hop_length
            # ax2.plot(diff_time + hop_length/2, std_diffs, color='purple', linestyle=':', label='Abs(Std Diff)')
            # ax2.axhline(y=diff_threshold, color='orange', linestyle='--', label='Diff Threshold')

            # 再次绘制检测到的异常区域
            for start_t, end_t in abnormal_segments:
                ax2.axvspan(start_t, end_t, color='red', alpha=0.3)

            ax2.set_title('Frame Standard Deviation Over Time')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Standard Deviation')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()

            plt.tight_layout()
            plt.show()

        return abnormal_segments

    # audio_waveform, sr = librosa.load("/Users/zhou/Downloads/test_uh.m4a", mono=True, sr=16000)
    # detect_constant_std_segments(audio_waveform, sr, plot_results=True)


class QiniuConst:
    access_key = "izz8Pq4VzTJbD8CmM3df5BAncyqynkPgF1K4srqP"
    secret_key = "pOhSAES6tocA3PzNF2fS_bnShTLUX5TEA1-tUmJY"
    bucket_domain = "http://resource.aisounda.cn"
    bucket_public_domain = "https://public.yisounda.com"
    bucket_name = 'sounda'
    bucket_public_name = 'sounda-public'


def post2qiniu(localfile, key, bkt=QiniuConst.bucket_name):
    os.path.exists(localfile)

    # >>> 上传
    q = Auth(QiniuConst.access_key, QiniuConst.secret_key)
    token = q.upload_token(bkt, key, 3600)  # 生成上传Token，可以指定token的过期时间等
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
