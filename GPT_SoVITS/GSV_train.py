
import os
import sys
from webui import open1abc
from subprocess import Popen,getstatusoutput
# PROJ_DIR = "~/AudioProject/GPT-SoVITS"
INPUT_DIR = "~/AudioProject/voice_sample/XiaoLinShuo"
EXP_NAME = "XiaoLine"
SLICE_DIR = os.path.join(INPUT_DIR, 'sliced')
DENOISED_DIR = os.path.join(INPUT_DIR, 'denoised')
ASR_DIR = os.path.join(INPUT_DIR, 'asr')
ASR_FP = os.path.join(ASR_DIR, os.path.basename(DENOISED_DIR))+".list"

status, output = getstatusoutput("ls tools")
assert status == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python GPT_SoVITS/GSV_train.py"


# 人声分离
# cmd = "demucs --two-stems=vocals xx"

# 切片
def step_slice():
    threshold = -34  # 音量小于这个值视作静音的备选切割点
    min_length = 4000  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
    min_interval = 100  # 最短切割间隔 (说话快的人就可以调小一点，默认是300）
    hop_size = 10  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
    max_sil_kept = 500  # 切完后静音最多留多长
    _max = 0.9  # 归一化后最大值多少
    alpha = 0.25  # 混多少比例归一化后音频进来
    n_parts = 4  # 并行数
    ps_slice = []
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = f"""
                python 'tools/slice_audio.py' \
                {INPUT_DIR} \
                {SLICE_DIR} \
                {threshold} {min_length} {min_interval} {hop_size} {max_sil_kept} {_max} {alpha} {i_part} {n_parts}
            """.strip()
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        for p in ps_slice:
            p.wait()
        ps_slice = []


# 降噪
def step_denoise():
    # todo 变成并行的
    cmd = f"""python 'tools/cmd-denoise.py' \
    -i {SLICE_DIR} \
    -o {DENOISED_DIR} \
    -p float32 \
    -m 'tools/denoise-model/speech_frcrn_ans_cirm_16k'
    """
    print(cmd)
    p_denoise = Popen(cmd, shell=True)
    p_denoise.wait()


# ASR
def step_asr(lang="zh"):
    from tools.asr.config import asr_dict
    asr_py = asr_dict["达摩 ASR (中文)"]["path"]
    if lang != "zh":
        asr_py = asr_dict["Faster Whisper (多语种)"]["path"]

    p_asr = None
    if (p_asr == None):
        cmd = f"""python 'tools/asr/{asr_py}' \
        -i {DENOISED_DIR} \
        -o {ASR_DIR} \
        -s large -l zh -p float32 \
        """
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None

# todo 用open1abc替换这三个
def step_speech_to_text():
    # webui.py open1a
    # python GPT_SoVITS/prepare_datasets/1-get-text.py
    cmd = f"""
    
    """
    pass

def step_ssl_extract():
    # open1b
    # python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
    pass


def step_semantic():
    # open1c
    # python GPT_SoVITS/prepare_datasets/3-get-semantic.py
    pass


def step_train_sovits():
    pass


def step_train_gpt():
    pass


if __name__ == '__main__':
    # step_slice()
    # step_denoise()
    # step_asr("zh")

    pass
