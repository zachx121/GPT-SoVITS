import numpy as np
import subprocess
from service_GSV.GSV_model import GSVModel
from service_GSV.GSV_const import Route as R
from service_GSV.GSV_const import ReferenceInfo
from service_GSV import GSV_const as C
import soundfile as sf
import re
import os
import librosa
import sys
from tqdm.auto import tqdm

SEP_MAP = {"zh_cn": ["。", "，", "？", "！", ",", "?", "!"],
           "en_us": [".", ",", "?", "!"]}

get_token_num = lambda t, tlang: len(t.split(" ")) if tlang == "en_us" else len(t)


def get_estimate_func(x, y, debug=False, rank=2):
    x, y = np.array(x), np.array(y)
    # 1. 去掉 x 和 y 中比值最大的两个数据和最小的两个数据
    # 计算 x 和 y 的比值
    ratios = y / x

    # 获取比值最大的两个数据的索引
    max_indices = np.argsort(ratios)[-2:]
    # 获取比值最小的两个数据的索引
    min_indices = np.argsort(ratios)[:2]

    # 合并要去除的索引
    remove_indices = np.concatenate((max_indices, min_indices))

    # 去除异常数据
    x_filtered = np.delete(x, remove_indices)
    y_filtered = np.delete(y, remove_indices)

    # 2. 用一个二次多项式来拟合 x 和 y 的关系
    # 进行二次多项式拟合
    coefficients = np.polyfit(x_filtered, y_filtered, rank)
    # a, b, c = coefficients

    # 生成拟合曲线的 x 值
    x_fit = np.linspace(min(x_filtered), max(x_filtered), 100)
    # 计算拟合曲线的 y 值
    y_fit = np.zeros_like(x_fit)
    for idx,p in enumerate(coefficients):
        y_fit = y_fit + p*np.power(x_fit,len(coefficients)-1-idx)
        # y_fit = a * x_fit**2 + b * x_fit + c

    # 打印拟合的系数
    print(f"使用 {rank}次多项式拟合，参数依次为: {coefficients}")
    if debug:
        from matplotlib import pyplot as plt
        # 绘制原始数据和拟合曲线
        plt.scatter(x_filtered, y_filtered, label='Filtered Data')
        plt.plot(x_fit, y_fit, 'r-', label='Quadratic Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Quadratic Fit of Filtered Data')
        plt.legend()
        plt.show()
    return coefficients


# python core/quality_check/estimate_audio_len_v2.py
if __name__ == '__main__':
    get_token_num = lambda t, tlang: len(t.split(" ")) if tlang in ["en", "en_us"] else len(t)
    # 英文素材 和 对应的SID（需要用这个SID在本地训练过，这样在voice_sample才会有对应的训练音频数据、ASR文本，用于拟合）
    LANG = "en"
    sid_list = [
        "ChatTTS_Voice_Clone_Common_PhenixV2",
        "ChatTTS_Voice_Clone_Common_OrionV2",
        "ChatTTS_Voice_Clone_Common_MarthaV2",
        "ChatTTS_Voice_Clone_Common_ZoeV2",
        "ChatTTS_Voice_Clone_Common_NinaV2",
        "ChatTTS_Voice_Clone_Common_Mike"
    ]

    # 中文素材
    # LANG = "zh"
    # sid_List = [
    #     "ChatTTS_Voice_Clone_User_3951_20250123010229136_x8bf"
    # ]

    x, y = [], []
    audio_list, text_list = [], []
    for sid in sid_list:
        fp = os.path.join(C.VOICE_SAMPLE_DIR, sid, "asr", "denoised.list")
        if not os.path.exists(fp):
            print(f"file not exist: '{fp}'")
            continue
        res = subprocess.run(["cat", fp], capture_output=True, text=True, encoding='utf-8')
        assert res.returncode == 0
        for line in res.stdout.split("\n"):
            audio_fp, _, lang, text = line.split("|")
            if not os.path.exists(audio_fp):
                print(f"audio_fp not exist: '{audio_fp}'")
                continue

            audio_list.append(audio_fp)
            text_list.append(text)
            x.append(get_token_num(text, LANG))
            y.append(librosa.get_duration(filename=audio_fp))

    a,b,c,d = get_estimate_func(x, y, debug=False, rank=4)
    print(f">>> 三次多项式参数: a, b, c, d=({a},{b},{c},{d})")

