import numpy as np
import subprocess
from service_GSV.GSV_model import GSVModel
from service_GSV.GSV_const import Route as R
from service_GSV.GSV_const import ReferenceInfo
from service_GSV import GSV_const as C
import soundfile as sf
import re
import os
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


# python core/quality_check/estimate_audio_len.py 'test_cxm' 'zh_cn'
# python core/quality_check/estimate_audio_len.py 'ChatTTS_Voice_Clone_Common_NinaV2' 'en_us'
# python core/quality_check/estimate_audio_len.py 'ChatTTS_Voice_Clone_Common_Mike' 'en_us'
if __name__ == '__main__':
    sid = sys.argv[1]
    lang = sys.argv[2]  # "zh_cn", "en_us"
    all_sep = "".join(SEP_MAP[lang])
    text_list = []
    text_fp = os.path.abspath(os.path.join(__file__, "..", f"text_{lang}.txt"))
    with open(text_fp, encoding="utf-8") as fr:
        for i in fr.readlines():
            for j in re.split(f"[{all_sep}]", i.strip("\n")):
                j = j.strip()
                if len(j) != 0:
                    text_list.append(j)
    print(f">>> 总计使用多少句话: {len(text_list)}")

    sovits_model = R.get_sovits_fp(sid)
    gpt_model = R.get_gpt_fp(sid)
    M = GSVModel(sovits_model_fp=sovits_model, gpt_model_fp=gpt_model)

    ref_audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    # ref_lang, ref_text = subprocess.getoutput(f"cat {R.get_ref_text_fp(sid, C.D_REF_SUFFIX)}").strip().split("|")
    res = subprocess.run(["cat", R.get_ref_text_fp(sid, C.D_REF_SUFFIX)], capture_output=True, text=True, encoding='utf-8')
    assert res.returncode == 0
    ref_lang, ref_text = res.stdout.strip().split("|")

    ref_info = ReferenceInfo(audio_fp=ref_audio_fp, lang=ref_lang, text=ref_text)

    x_list, y_list = [], []
    audios, sr = [], -1
    for text in tqdm(text_list):
        wav_sr, wav_arr_int16, _ = M.predict(target_text=text,
                                             target_lang=lang,
                                             ref_info=ref_info,
                                             top_k=30, top_p=0.99, temperature=0.3,
                                             ref_free=False, no_cut=False)
        audios.append(wav_arr_int16)
        sr = wav_sr
        num_tokens = get_token_num(text, lang)
        audio_duration = wav_arr_int16.shape[0] / wav_sr
        x_list.append(num_tokens)
        y_list.append(audio_duration)

    a, b, c, d = get_estimate_func(x_list, y_list, rank=3)
    print(f">>> 基于sid={sid}拟合，按字符数x,预估音频时长y=a*(x)^2+b*x+c")

    print(f">>> 识别到不合理的音频有: ")
    for idx, audio in enumerate(audios):
        x = get_token_num(text_list[idx], lang)
        y = a*x**3 + b*x**2 + c*x + d
        p = audio.shape[0] / sr
        if abs(y - p) >= 3:
            print(f"    实际耗时:{p:.4f} 预估耗时:{y:.4f} 文本:{text[idx]} ")
            sf.write(f"./tmp_output/out_{idx}_x{x}_y{y}_p{p}.wav", audio, sr)

    print(f">>> 基于sid={sid}合成的音频，拟合了{len(x_list)}个样本，记字符数x,预估音频时长y=a*(x)^2+b*x+c")
    print(f"    目前英文是空格分词后计算token个数，中文是直接字符数")
    print(f">>> 二次多项式参数: a,b,c,d=({a},{b},{c},{d})")
