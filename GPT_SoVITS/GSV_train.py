import logging
from logging.handlers import TimedRotatingFileHandler

import os
import shutil
import sys
import json
import yaml
import wave
import re
import time
import base64
import socket
import traceback
import multiprocessing as mp

from urllib.parse import unquote
from subprocess import Popen, getstatusoutput

import numpy as np
import librosa
import torch
from flask import Flask, request

import requests as http_requests  # 给 requests 库设置别名

import utils_audio
import GSV_const as C  # 使用已有的常量配置
from GSV_const import Route as R
from GSV_model import GSVModel, ReferenceInfo
import pika

assert getstatusoutput("ls tools")[0] == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python GPT_SoVITS/GSV_train.py"


# 人声分离
# cmd = "demucs --two-stems=vocals xx"

def _get_duration_of_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / rate
    return duration


def get_latest_fp(inp_dir):
    _inp_dir = os.path.expanduser(inp_dir)
    fp = sorted(os.listdir(_inp_dir),
                key=lambda x: os.path.getmtime(os.path.join(_inp_dir, x)), reverse=True)[0]
    return fp


def step_convert2wav(inp):
    for name in sorted(list(os.listdir(inp))):
        if any(name.endswith(i) for i in ['m4a', 'mp3', 'mp4']):
            # inp = "/root/GPT-SoVITS/voice_sample/ChatTTS_Voice_Clone_4_222rb2j"
            fp = os.path.join(inp, name)
            new_fp = os.path.join(inp, os.path.splitext(name)[0] + ".wav")
            cmd = f"ffmpeg -y -i {fp} {new_fp}"
            logger.info(f">>> convert2wav: '{cmd}'")
            s, _ = getstatusoutput(cmd)
            assert s == 0, f"ffmpeg转换格式时错误, cmd: '{cmd}'"


# 切片 min_interval常规还是用100，语速快的用80
def step_slice(inp_dir, out_dir, min_interval=100):
    threshold = -34  # 音量小于这个值视作静音的备选切割点
    min_length = 4000  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
    # min_interval = 100  # 最短切割间隔 (说话快的人就可以调小一点，默认是300）
    hop_size = 10  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
    max_sil_kept = 500  # 切完后静音最多留多长
    _max = 0.9  # 归一化后最大值多少
    alpha = 0.25  # 混多少比例归一化后音频进来
    n_parts = 1  # 并行数
    ps_slice = []
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = f"""
                python 'tools/slice_audio.py' \
                {inp_dir} \
                {out_dir} \
                {threshold} {min_length} {min_interval} {hop_size} {max_sil_kept} {_max} {alpha} {i_part} {n_parts}
            """.strip()
            logger.info(f"execute cmd: {cmd}")
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        for p in ps_slice:
            p.wait()
        ps_slice = []

    # 计算切分后平均音频文件的时长
    logger.info(">>> Slice finished. ")
    total_duration, max_duration, num = 0, 0, 0
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                duration = _get_duration_of_wav(file_path)
                max_duration = max(max_duration, duration)
                total_duration += duration
                num += 1
    logger.info(f">>> Slice finished. output wav: [Num]:{num} [AvgDuration]:{total_duration/num:.04f}, [MaxDuration]:{max_duration}")


# 降噪
def step_denoise(slice_dir, denoised_dir):
    # todo 变成并行的
    cmd = f"""python 'tools/cmd-denoise.py' \
    -i {slice_dir} \
    -o {denoised_dir} \
    -p float32 \
    -m 'tools/denoise-model/speech_frcrn_ans_cirm_16k'
    """
    print(cmd)
    p_denoise = Popen(cmd, shell=True)
    p_denoise.wait()


# ASR
def step_asr(denoised_dir, asr_dir, lang="auto"):
    from tools.asr.config import asr_dict
    asr_py = asr_dict["达摩 ASR (中文)"]["path"]
    if lang.upper() != "ZH":
        asr_py = asr_dict["Faster Whisper (多语种)"]["path"]
    logger.info(f">>> Step_ASR's asr_py is '{asr_py}'")

    p_asr = None
    if (p_asr == None):
        # python tools/asr/fasterwhisper_asr.py -i ./voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz/denoised -o ./voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz/asr -s large -l en -p float32
        cmd = f"""python 'tools/asr/{asr_py}' \
        --input_folder {denoised_dir} \
        --output_folder {asr_dir} \
        --model_size large-v3-local \
        --language {lang.lower()} -p float32 \
        """
        logger.info(f"asr cmd: {cmd}")
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None


    fp = os.path.join(asr_dir, "denoised.list")
    with open(fp, "r") as fpr:
        lines = fpr.readlines()
        total_words = sum([len(l.split("|")[3].split(" ")) for l in lines])
        longest = max([len(l.split("|")[3].split(" ")) for l in lines])
        avg_words = total_words/len(lines)
    logger.info(f">>> ASR Finished. [Lines]: {len(lines)} [AvgWords]: {avg_words:.02f} [LongestSeg]: {longest}")


# 不能直接用webui.py里的open1abc，因为那个函数返回里用了yield
def step_apply_pretrains(asr_fp, exp_root_dir, is_half,sid):
    logger.info(f">>> Apply Pretrains on '{asr_fp}'... ")
    inp_text = asr_fp
    exp_name = sid
    opt_dir = "%s/%s" % (exp_root_dir, exp_name)
    inp_wav_dir = ""  # 直接使用denoised.list里的绝对路径
    gpu_numbers1a = "0-0"
    gpu_numbers1Ba = "0-0"
    gpu_numbers1c = "0-0"
    bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ssl_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    pretrained_s2G_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"

    ps1abc = []
    ###########################
    # 1a
    ###########################
    path_text = "%s/2-name2text.txt" % opt_dir
    if (os.path.exists(path_text) == False or (os.path.exists(path_text) == True and len(
            open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2)):
        logger.info(f"writing to {path_text}")
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names = gpu_numbers1a.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = 'python GPT_SoVITS/prepare_datasets/1-get-text.py'
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1abc.append(p)

        for p in ps1abc: p.wait()

        opt = []
        for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
        logger.info(f"    1a-done. result in '{path_text}'")
    else:
        logger.info("    1a skipped.")

    ps1abc = []
    ###########################
    # 1b
    ###########################
    config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "cnhubert_base_dir": ssl_pretrained_dir,
    }
    gpu_names = gpu_numbers1Ba.split("-")
    all_parts = len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
            }
        )
        os.environ.update(config)
        cmd = 'python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1abc.append(p)
    logger.info("    1a-done. 1b-ING")
    for p in ps1abc: p.wait()
    ps1abc = []
    logger.info("    1a1b-done.")
    ###########################
    # 1c
    ###########################
    path_semantic = "%s/6-name2semantic.tsv" % opt_dir
    if (os.path.exists(path_semantic) == False or (
            os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31)):
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": "GPT_SoVITS/configs/s2.json",
        }
        gpu_names = gpu_numbers1c.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = 'python GPT_SoVITS/prepare_datasets/3-get-semantic.py'
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1abc.append(p)
        logger.info("    1a1b-done. 1c-ING")
        for p in ps1abc: p.wait()

        opt = ["item_name\tsemantic_audio"]
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        logger.info("1a1b1c-done. All Done.")
    else:
        logger.info("1c skipped.")
    ps1abc = []

    logger.info(f"Pretrain抽取特征完毕，可检查输出目录 {os.path.abspath(os.path.join('logs', sid))}")


def step_train_sovits(sid, total_epoch, is_half, exp_root_dir, sovits_weight_root, tmp_dir):
    batch_size = 18
    exp_name = sid
    text_low_lr_rate = 0.4
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 4
    gpu_numbers1Ba = "0"
    pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"

    with open("GPT_SoVITS/configs/s2.json") as f:
        data = f.read()
        data = json.loads(data)

    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers1Ba
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = os.path.join(exp_root_dir, sid)
    data["save_weight_dir"] = sovits_weight_root
    data["name"] = exp_name
    tmp_config_path = os.path.join(tmp_dir, f"s2_{sid}.json")
    with open(tmp_config_path, "w") as f: f.write(json.dumps(data))

    # logs_s2用来存放Generator和Discriminator模型的
    os.makedirs(os.path.join(exp_root_dir, sid, "logs_s2"), exist_ok=True)
    cmd = f'python GPT_SoVITS/s2_train.py --config "{tmp_config_path}"'
    logger.info(cmd)
    p_train_SoVITS = Popen(cmd, shell=True)
    p_train_SoVITS.wait()
    p_train_SoVITS = None


def step_train_gpt(sid, total_epoch, is_half, exp_root_dir, gpt_weight_root, tmp_dir):   
    batch_size = 18
    exp_name = sid
    if_dpo = False
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 5
    gpu_numbers = "0"
    pretrained_s1 = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

    with open("GPT_SoVITS/configs/s1longer.yaml") as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    s1_dir = os.path.join(exp_root_dir, sid)
    os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
    if (is_half == False):
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = gpt_weight_root
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1" % s1_dir

    os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")
    os.environ["hz"] = "25hz"
    tmp_config_path = os.path.join(tmp_dir, f"s1_{sid}.yaml")
    with open(tmp_config_path, "w") as f: f.write(yaml.dump(data, default_flow_style=False))
    # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
    cmd = f'python GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}" '
    print(cmd)
    p_train_GPT = Popen(cmd, shell=True)
    p_train_GPT.wait()
    p_train_GPT = None


# 优化后的辅助函数
def ensure_dir_exists(directory):
    """ 确保目录存在，如果存在则清理掉 """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def download_file(url, target_dir):
    """ 下载文件并保存到指定目录 """
    filename = os.path.basename(unquote(url).split("?")[0])
    file_path = os.path.join(target_dir, filename)
    try:
        response = http_requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.error(f"Failed to download {url}, status code {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")


def download_files_in_parallel(urls, target_dir, num_workers=4):
    """ 并行下载多个文件 """
    with mp.Pool(num_workers) as pool:
        pool.starmap(download_file, [(url, target_dir) for url in urls])


def log_audio_statistics(asr_dir):
    """ 打印音频数据的统计信息 """
    with open(os.path.join(asr_dir, "denoised.list"), "r") as fpr:
        lines = fpr.readlines()
    
    total_lines = len(lines)
    avg_words = sum([len(line.split("|")[3].split(" ")) for line in lines]) / total_lines
    max_words = max([len(line.split("|")[3].split(" ")) for line in lines])
    avg_duration = sum([_get_duration_of_wav(line.split("|")[0]) for line in lines]) / total_lines
    max_duration = max([_get_duration_of_wav(line.split("|")[0]) for line in lines])

    logger.info(f"""
        >>> 整体数据处理结果
        [total lines]: {total_lines}
        [avg words per line]: {avg_words}
        [max words per line]: {max_words}
        [avg audio duration]: {avg_duration}
        [max audio duration]: {max_duration}
    """)


def send_result_to_queue(channel, result):
    """ 发送训练结果到队列 """
    channel.basic_publish(
        exchange='',
        routing_key="queue_tran_result",
        body=json.dumps(result),  # 确保结果被序列化为 JSON
        properties=pika.BasicProperties(content_type='application/json')
    )
    logger.info(f"Training result sent to tran_result_queue: {result}")


def clean_temp_dirs(dirs):
    """ 清理临时文件夹 """
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            

def train_model(task):
    """
    执行数据准备和模型训练
    """

    sid = task['speaker']
    LANG = C.LANG_MAP[task['lang']]
    data_urls = task['data_urls']
    post2oss = "1"

    INPUT_DIR = os.path.join(C.VOICE_SAMPLE_DIR, sid)  # 使用 C 中的 VOICE_SAMPLE_DIR
    ensure_dir_exists(INPUT_DIR)

    logger.info(f">>> Start Data Preparing.")
    download_files_in_parallel(data_urls, INPUT_DIR)

    if len(os.listdir(INPUT_DIR)) == 0:
        raise Exception("All Audio URLs failed to download.")

    logger.info(f">>> Start Model Training.")
    IS_HALF = False
    SLICE_DIR = os.path.join(INPUT_DIR, 'sliced')
    DENOISED_DIR = os.path.join(INPUT_DIR, 'denoised')
    ASR_DIR = os.path.join(INPUT_DIR, 'asr')
    ASR_FP = os.path.join(ASR_DIR, os.path.basename(DENOISED_DIR)) + ".list"
    EXP_ROOT_DIR = C.LOG_DIR
    TMP_DIR = os.path.join(EXP_ROOT_DIR, "TEMP_CONFIG")
    SoVITS_weight_root = os.path.join(C.SOVITS_DIR, sid)
    GPT_weight_root = os.path.join(C.GPT_DIR, sid)

    logger.info(f">>> Start with ExpName='{sid}', InputDir='{INPUT_DIR}', Language='{LANG}'")
    
    # 清理上次遗留的数据
    for dir_path in [SLICE_DIR, DENOISED_DIR, ASR_DIR, EXP_ROOT_DIR, TMP_DIR, SoVITS_weight_root, GPT_weight_root]:
        ensure_dir_exists(dir_path)

    logger.info(">>> At step_convert2wav")
    step_convert2wav(INPUT_DIR)
    logger.info(">>> At step_slice")
    step_slice(INPUT_DIR, SLICE_DIR, min_interval=80 if LANG == "EN" else 100)
    logger.info(">>> At step_denoise")
    step_denoise(SLICE_DIR,DENOISED_DIR)
    logger.info(">>> At step_asr")
    step_asr(DENOISED_DIR,ASR_DIR,LANG)

    log_audio_statistics(ASR_DIR)

    logger.info(">>> At step_apply_pretrains")
    step_apply_pretrains(ASR_FP,EXP_ROOT_DIR,IS_HALF,sid)
    logger.info(">>> At step_train_sovits")
    step_train_sovits(sid,8,IS_HALF,EXP_ROOT_DIR,SoVITS_weight_root,TMP_DIR)
    logger.info(">>> At step_train_gpt")
    step_train_gpt(sid,15,IS_HALF,EXP_ROOT_DIR,GPT_weight_root,TMP_DIR)

    # 重命名模型文件
    sovits_fp = os.path.join(SoVITS_weight_root, sid + ".latest.pth")
    gpt_fp = os.path.join(GPT_weight_root, sid + ".latest.ckpt")
    os.rename(os.path.join(SoVITS_weight_root, get_latest_fp(SoVITS_weight_root)), sovits_fp)
    os.rename(os.path.join(GPT_weight_root, get_latest_fp(GPT_weight_root)), gpt_fp)

    if post2oss == "1":
        logger.info(">>> Uploading sovits model to qiniu.")
        url = utils_audio.post2qiniu(sovits_fp, R.get_sovits_osskey(sid))
        logger.info(f">>> url as: '{url}'")
        logger.info(">>> Uploading gpt model to qiniu.")
        url = utils_audio.post2qiniu(gpt_fp, R.get_gpt_osskey(sid))
        logger.info(f">>> url as: '{url}'")
        
    logging.info(">>> Uploading default_ref_audio to qiniu.")
    asr_fp = os.path.join(C.VOICE_SAMPLE_DIR, sid, "asr", "denoised.list")
    with open(asr_fp, "r", encoding='utf-8') as fr:
        line = fr.readline().strip()
    infos = line.split("|")
    _audio_fp = infos[0]
    _lang = infos[2]
    _lang_map = {v: k for k, v in C.LANG_MAP.items()}
    _lang = _lang_map[_lang]
    _text = infos[3]
    with open(R.get_ref_text_fp(sid, C.D_REF_SUFFIX), "w") as fpw:
        fpw.write(f"{_lang}|{_text}")
    audio_url = utils_audio.post2qiniu(_audio_fp, R.get_ref_audio_osskey(sid))
    text_url = utils_audio.post2qiniu(R.get_ref_text_fp(sid, C.D_REF_SUFFIX), R.get_ref_text_osskey(sid))
    logging.info(f">>> [audio_url]:'{audio_url}' [text_url]:'{text_url}'")
    
    # Show models path
    models_fp = []
    for i in os.listdir(SoVITS_weight_root):
        if sid in i:
            models_fp.append(os.path.abspath(os.path.join(SoVITS_weight_root, i)))
    for i in os.listdir(GPT_weight_root):
        if sid in i:
            models_fp.append(os.path.abspath(os.path.join(GPT_weight_root, i)))
    logger.info(">>> Models path:\n%s" % "\n".join(models_fp))
    logger.info("<<< Training finished.")

    result= {"code": 0, "msg": "Model Training finish.", "result": sid}


    # 直接重新建立连接，因为tran耗时比较久，这时候断开了
    connection, channel = connect_to_rabbitmq()
    # 发送结果到队列
    send_result_with_retry(channel, result)

    # 清理临时文件
    clean_temp_dirs([SLICE_DIR, DENOISED_DIR, TMP_DIR])


queue_tran_request = "queue_tran_request"
queue_tran_result = "queue_tran_result"


def train_consumer():
    connection, channel = connect_to_rabbitmq()

    while True:
        try:
            try:
                if channel is None or channel.is_closed:
                    connection, channel = connect_to_rabbitmq()
            except Exception:
                # 如果 抛出异常，直接重连
                logger.info(f"mq connect error,reconnect")
                connection, channel = connect_to_rabbitmq()
    
            method_frame, header_frame, body = channel.basic_get(queue=queue_tran_request, auto_ack=True)
            if body is None:
                time.sleep(0.1)  # 如果没有消息，休眠一段时间
                continue  # 如果没有消息，等待下一次
            # 解析任务消息
            task = json.loads(body)
            logger.info(f"Received training task: {task}")

            # 执行训练逻辑
            train_model(task)

        except pika.exceptions.AMQPConnectionError:
            logger.error("Connection to RabbitMQ lost, attempting to reconnect...")
            connection, channel = connect_to_rabbitmq()
        except Exception as e:
            logger.error(f"Error in task: {e}", exc_info=True)
            result = {"code": 1, "msg": "Model Training failed.", "result": task.get('speaker', 'unknown')}
            # 发送结果到队列
            send_result_with_retry(channel, result)
    

def send_result_with_retry(channel, result):
    for attempt in range(5):
        try:
            channel.basic_publish(
                exchange='',
                routing_key=queue_tran_result,
                body=json.dumps(result),
                properties=PROPERTIES  # Make message persistent
            )
            logger.info("send message to queue")

            break  # 发送成功，退出循环
        except pika.exceptions.ChannelClosed:
            logger.error("Channel closed, reconnecting...")
            connection, channel = connect_to_rabbitmq()  # 重新连接
        except Exception as e:
            logger.error(f"Failed to send result: {e}")
            time.sleep(2)  # 等待后重试


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        "virtual_host": "device-public"
    }

    # 连接到 RabbitMQ
    credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config["address"],
        port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
        virtual_host=rabbitmq_config["virtual_host"],
        credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")
        return None, None
    

if __name__ == "__main__":
    try:
        # 获取实例编号（从启动参数中获取）
        if len(sys.argv) > 1:
            instance_id = sys.argv[1]  # 启动时传入的实例编号
        else:
            instance_id = "default"  # 如果未传入参数，使用默认值

        # 日志目录（代码中自动创建，无需脚本管理）
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
        else:
            print(f"Log directory already exists: {log_dir}")

        # 根据实例编号生成日志文件名
        log_file = f"consumer-{instance_id}.log"
        log_path = os.path.join(log_dir, log_file)

        # 配置日志
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        file_handler = TimedRotatingFileHandler(
        filename=log_path,  # 日志文件路径
        when="midnight",    # 按天分隔（午夜生成新日志文件）
        interval=1,         # 每 1 天分隔一次
        backupCount=7,      # 最多保留最近 7 天的日志文件
        encoding="utf-8"    # 设置编码，避免中文日志乱码
        )
        file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
        file_handler.setFormatter(logging.Formatter(
            fmt='[%(asctime)s-%(levelname)s]: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        # 将文件处理器添加到日志记录器中
        logger.addHandler(file_handler)

        train_consumer()
    except KeyboardInterrupt:
        logger.info("Consumer stopped.")