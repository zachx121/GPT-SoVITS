"""
[route]: /train_model
[json-params]: {speaker:"..", lang:"zh", data_urls:["...", "..." ]}

[route]: /model_status
[json-params]: {speaker_list: ["..", ".."]}


[route]: /add_reference
[json-params]:
{"speaker": "..",
 "ref_audio_url": "...",
 "ref_text_url": "...", 注意文本格式文本是 "zh_cn|你好我是四郎" 这样的，即按竖线分割的，前面是语言后面是音频的文本
 "ref_suffix":".." optional 可以直接不传，传了的话后面推理接口也要传。当可以提供多个参考音频时，可以用情绪作为后缀}

[route]: /init_model
[json-params]: {speaker:"..."}

[route]: /inference
[json-params]:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. "JP", "ZH", "EN", "ZH_EN", "JP_EN")
    use_ref: bool = True  # 推理时是否使用参考音频的情绪
    ref_suffix: str = D_REF_SUFFIX  # 可不传，用于指定参考音频的后缀
"""

import re
import librosa
import torch
import time
import numpy as np
import os
os.environ['TQDM_DISABLE'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import base64
import json
import shutil
from urllib.parse import unquote
from subprocess import getstatusoutput, check_output
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from GSV_model import GSVModel, ReferenceInfo
import GSV_const as C
from GSV_const import Route as R
import multiprocessing as mp
import utils_audio

app = Flask(__name__, static_folder="./static_folder", static_url_path="")


def model_process(sid, q_inp, q_out, event):
    M = GSVModel(sovits_model_fp=R.get_sovits_fp(sid),
                 gpt_model_fp=R.get_gpt_fp(sid),
                 speaker=sid)
    event.set()
    while True:
        try:
            p: C.InferenceParam = q_inp.get()
            tlist = []
            tlist.append(int(time.time()*1000))
            if p is None:
                break
            tlist.append(int(time.time() * 1000))
            wav_sr, wav_arr_int16 = M.predict(target_text=p.text,
                                              target_lang=p.tgt_lang,
                                              ref_info=p.ref_info,
                                              top_k=1, top_p=0, temperature=0,
                                              ref_free=p.ref_free, no_cut=p.nocut)
            tlist.append(int(time.time()*1000))
            wav_arr_float32 = wav_arr_int16.astype(np.float32) / 32768.0
            wav_arr_float32_16khz = librosa.resample(wav_arr_float32, orig_sr=wav_sr, target_sr=16000)
            wav_arr_int16 = (np.clip(wav_arr_float32_16khz, -1.0, 1.0) * 32767).astype(np.int16)
            tlist.append(int(time.time()*1000))
            rsp = {"trace_id": p.trace_id,
                   # "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
                   "audio_buffer_int16": base64.b64encode(wav_arr_int16.tobytes()).decode(),
                   "sample_rate": 16000,
                   "code": 0,
                   "msg": "success."}
            rsp = json.dumps({"code": 0,
                              "msg": "",
                              "result": rsp})
            tlist.append(int(time.time()*1000))
            # sf.write(f"output_{time.time():.0f}.wav", wav_arr_int16, wav_sr)
            tlist.append(int(time.time()*1000))
            q_out.put(rsp)
            tlist.append(int(time.time()*1000))
            print(f"model time tlist: " + ",".join([f'{b - a}ms' for a, b in zip(tlist[:-1], tlist[1:])]))
        except Exception as e:
            logging.error(f">>> Error when model.predict. e: {repr(e.message)}")
    # 结束时清理掉模型和显存
    del M
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# 返回所有GPU的内存空余量，是一个list
def get_free_gpu_mem():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


@app.route("/load_model", methods=['POST'])
def load_model():
    """
    http params:
    - speaker:str
    - speaker_num:int
    """
    res = {"code": 0, "msg": "", "result": ""}
    info = request.get_json()
    sid = info['speaker']
    sid_num = info['speaker_num']
    download_overwrite = info.get("download_overwrite", "1")
    logging.debug(f"load_model: {info}")
    if sid in M_dict:
        # 直接全部unload重新加载，减少逻辑
        # if sid_num == len(M_dict[sid]['process_list']):
        res['status'] = 1
        cur_num = len(M_dict[sid]['process_list'])
        res['msg'] = f"Already Init ({sid}_x{cur_num}). call `unload_model` first."
        return json.dumps(res)

    # 假设一个模型占用2.8GB显存
    if get_free_gpu_mem()[0] <= 2800 * (sid_num+1):
        res['status'] = 1
        res['msg'] = 'GPU OOM'
        return json.dumps(res)

    # 从OSS上加载模型
    if download_overwrite == "1" or (not (os.path.exists(R.get_sovits_fp(sid)) and os.path.exists(R.get_gpt_fp(sid)))):
        try:
            utils_audio.download_from_qiniu(R.get_sovits_osskey(sid), R.get_sovits_fp(sid))
            utils_audio.download_from_qiniu(R.get_gpt_osskey(sid), R.get_gpt_fp(sid))
        except Exception as e:
            logging.error(f"error when download '{sid}': {repr(e.message)}")
            res['status'] = 1
            res['msg'] = f"model of '{sid}' is not found and download failed"
            return json.dumps(res)

    # 开启N个子进程加载模型并等待Queue里的数据来处理请求
    q_inp = mp.Queue()
    q_out = mp.Queue()
    process_list = []
    _load_events = []
    for _ in range(sid_num):
        event = mp.Event()
        p = mp.Process(target=model_process, args=(sid, q_inp, q_out, event))
        process_list.append(p)
        _load_events.append(event)
        p.start()
    # 阻塞直到所有模型加载完毕
    while not all([event.is_set() for event in _load_events]):
        pass
    M_dict[sid] = {"q_inp": q_inp, "q_out": q_out, "process_list": process_list}
    res['msg'] = "Init Success"
    logging.info(f"<<< Init of '{sid}' finished.")
    return json.dumps(res)


@app.route("/unload_model", methods=['POST'])
def unload_model():
    info = request.get_json()
    sid = info["speaker"]
    if sid not in M_dict:
        res = {"code": 1,
               "msg": f"sid('{sid}') not loaded yet",
               "result": ""}
        return res

    # 有N个子进程，所以给队列放入N个None确保每个子进程全都能收到结束信号
    for _ in range(len(M_dict[sid]["process_list"])):
        M_dict[sid]['q_inp'].put(None)
    for p in M_dict[sid]["process_list"]:
        p.join()
    # 清理掉字典里记录的kv
    del M_dict[sid]
    res = {"code": 0,
           "msg": "Model Unloaded",
           "result": ""}
    return json.dumps(res)


def add_default_ref(sid):
    asr_fp = os.path.join(C.VOICE_SAMPLE_DIR, sid, "asr", "denoised.list")
    os.makedirs(os.path.join(C.VOICE_SAMPLE_DIR, sid), exist_ok=True)
    audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    text_fp = R.get_ref_text_fp(sid, C.D_REF_SUFFIX)

    assert os.path.exists(asr_fp)
    with open(asr_fp, "r", encoding='utf-8') as fr:
        # 示例如下：
        # voice_sample/test_silang1636/denoised/ref_audio_default.wav_0000000000_0000127680.wav|denoised|ZH|真是脚带帽在头顶靴上下不？
        line = fr.readline().strip()

    infos = line.split("|")
    _audio_fp = infos[0]
    _lang = infos[2]
    _lang_map = {v: k for k, v in C.LANG_MAP.items()}
    _lang = _lang_map[_lang]
    _text = infos[3]

    cmd1 = f"cp {_audio_fp} {audio_fp}"
    s1, _ = getstatusoutput(cmd1)
    assert s1 == 0
    cmd2 = f"echo '{_lang}|{_text}' > {text_fp}"
    s2, _ = getstatusoutput(cmd2)
    assert s2 == 0


@app.route("/add_reference", methods=['POST'])
def add_reference():
    """
    如果只传speaker表示从数据集选第一个作为reference
    http params:
    - speaker:str 必传
    - ref_suffix:str optional
    - ref_audio_url:str optional
    - ref_text_url:str optional
    """
    info = request.get_json()
    logging.debug(info)
    sid = info['speaker']
    suffix = info.get("ref_suffix", C.D_REF_SUFFIX)
    os.makedirs(os.path.join(C.VOICE_SAMPLE_DIR, sid), exist_ok=True)
    audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    text_fp = R.get_ref_text_fp(sid, C.D_REF_SUFFIX)

    # 如果没有用默认的就说明有指定的音频，否则用训练集里的一句话
    if suffix != C.D_REF_SUFFIX:
        logging.info(f">>> will use **ASSIGNED** audio&text with suffix: '{suffix}'")
        cmd = f"wget \"{info['ref_audio_url']}\" -O {audio_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"code": 1,
                              "msg": f"Reference Audio Download Wrong",
                              "result": ""})
            return res

        cmd = f"wget \"{info['ref_text_url']}\" -O {text_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"code": 1,
                              "msg": f"Reference Text Download Wrong",
                              "result": ""})
            return res

        res = json.dumps({"code": 0,
                          "msg": "Reference added.",
                          "result": ""})
    else:
        add_default_ref(sid)
        res = json.dumps({"code": 0,
                          "msg": "Automatic Reference added.",
                          "result": ""})
    return res


@app.route("/inference", methods=['POST'])
def inference():
    """
    http params:
    - 以Param类的成员变量为准
    """
    tlist = []
    tlist.append(int(time.time()*1000))
    info = request.get_json()
    logging.warning(f"inference at {time.time():.03f}s info:{info}")
    p = C.InferenceParam(info)
    # 检查speaker是否已经加载
    if p.speaker not in M_dict:
        return json.dumps({"code": 1,
                           "msg": f"inference使用的角色音({p.speaker})未被加载。已加载角色音: {M_dict.keys()}",
                           "result": ""})
    # 检查是否设置过默认的ref
    ref_audio_fp = R.get_ref_audio_fp(p.speaker, C.D_REF_SUFFIX)
    ref_text_fp = R.get_ref_text_fp(p.speaker, C.D_REF_SUFFIX)
    if not (os.path.exists(ref_audio_fp) and os.path.exists(ref_text_fp)):
        logging.warning(f">>> reference as '{p.ref_suffix}' is not ready, init a default one from training data.")
        add_default_ref(p.speaker)
    tlist.append(int(time.time()*1000))
    M_dict[p.speaker]["q_inp"].put(p)
    tlist.append(int(time.time()*1000))
    result = M_dict[p.speaker]["q_out"].get()
    tlist.append(int(time.time()*1000))
    print(f"server api tlist: " + ",".join([f'{b - a}ms' for a, b in zip(tlist[:-1], tlist[1:])]))
    return result


@app.route("/train_model", methods=['POST'])
def train_model():
    """
    http params:
    - speaker: str
    - lang: str
    - data_urls: list[str]
    """
    info = request.get_json()
    logging.debug(info)
    sid = info['speaker']
    lang = info['lang']
    data_urls = info['data_urls']
    post2oss = "1"
    data_dir = os.path.join(C.VOICE_SAMPLE_DIR, sid)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    logging.info(f">>> Start Data Preparing.")
    for url in data_urls:
        logging.info(f">>> Downloading Sample from {url}")
        filename = os.path.basename(unquote(url).split("?")[0])
        cmd = f"wget \"{url}\" -O {os.path.join(data_dir,filename)}"
        logging.debug(cmd)
        status, output = getstatusoutput(cmd)
        if status != 0:
            logging.error(f"    Download fail. url is {url}")
    if len(os.listdir(data_dir)) == 0:
        res = json.dumps({"code": 1,
                          "msg": "All Audio url failed to download.",
                          "result": ""})
        return res
    logging.info(f">>> Start Model Training.")
    # nohup python GPT_SoVITS/GSV_train.py zh_cn ChatTTS_Voice_Clone_4_222rb2j voice_sample/ChatTTS_Voice_Clone_4_222rb2j > ChatTTS_Voice_Clone_4_222rb2j.train 2>&1 &
    # python GPT_SoVITS/GSV_train.py en_us ChatTTS_Voice_Clone_0_Mike_yvmz voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz 1 > ChatTTS_Voice_Clone_0_Mike_yvmz.train
    cmd = f"nohup python GPT_SoVITS/GSV_train.py {C.LANG_MAP[lang]} {sid} {data_dir} {post2oss} > {sid}.train 2>&1 &"
    logging.info(f"    cmd is {cmd}")
    status, output = getstatusoutput(cmd)
    if status != 0:
        logging.error(f"    Model training failed. error:\n{output}")
        res = json.dumps({"code": 1,
                          "msg": "Model Training Failed",
                          "result": ""})
    else:
        res = json.dumps({"code": 0,
                          "msg": "Model Training Started.",
                          "result": ""})
    return res


@app.route("/check_training_status", methods=['POST'])
def check_training_status():
    # 检查本机执行了几个模型训练
    cmd = "ps -ef | grep 'python GPT_SoVITS/GSV_train.py' | grep -v 'grep'"
    status, output = getstatusoutput(cmd)
    # 注意没有训练进程时，status是1，output是空串
    if output == "":
        res = json.dumps({"code": 0,
                          "msg": "success",
                          "result": []})
    else:
        assert status == 0, f"cmd execution failed. cmd:'{cmd}' output:'{output}'"
        # sid_list = [re.split("\s+", i)[10] for i in output.split("\n")]
        sid_list = [re.split(r"\s+", i)[10] for i in output.split("\n") if i]  # 确保只处理非空行
        res = json.dumps({"code": 0,
                          "msg": "success",
                          "result": sid_list})
    return res


@app.route("/model_status", methods=['POST'])
def model_status():
    # 检查本服务加载了那些speaker模型
    res = []
    for k, v in M_dict.items():
        p_list = v['process_list']
        size = len(p_list) if all([p.is_alive() for p in p_list]) else -1
        res.append({"model_name": k, "model_num": size})

    logging.debug(str(res))
    res = json.dumps({"code": 0,
                      "msg": "",
                      "result": res})
    return res


@app.route("/download_model", methods=['POST'])
def download_model():
    info = request.get_json()
    sid_list = info['speaker_list']
    res = []
    for sid in sid_list:
        try:
            utils_audio.download_from_qiniu(sid+"_sovits", R.get_sovits_fp(sid))
            utils_audio.download_from_qiniu(sid+"_gpt", R.get_gpt_fp(sid))
            res.append({"model_name": sid, "download_success": True})
        except Exception as e:
            logging.error(f"error when download '{sid}': {repr(e.message)}")
            res.append({"model_name": sid, "download_success": False})

    res = json.dumps({"code": 0,
                      "msg": "",
                      "result": res})
    return res


# OSS下载完模型后，要用这个检查是否下载成功
@app.route("/is_model_oss_available", methods=['POST'])
def is_model_oss_available():
    """
    模型训练是否完成
    http params:
    - speaker_list: list[str]
    """
    info = request.get_json()
    speaker_list = info['speaker_list']
    if "speaker_list" not in info:
        return json.dumps({"code": 1, "msg": "speaker_list is required", "result": []})
    m_status = []
    for sid in speaker_list:
        cond1 = utils_audio.check_on_qiniu(R.get_sovits_osskey(sid))
        cond2 = utils_audio.check_on_qiniu(R.get_gpt_osskey(sid))
        # cond1 = os.path.exists(R.get_sovits_fp(sid))
        # cond2 = os.path.exists(R.get_gpt_fp(sid))
        m_status.append({"model_name": sid, "is_available": cond1 and cond2})
    return json.dumps({"code": 0, "msg": "", "result": m_status})


if __name__ == '__main__':
    mp.set_start_method("spawn")
    logging.info("Preparing")
    os.makedirs(C.VOICE_SAMPLE_DIR, exist_ok=True)
    os.makedirs(C.GPT_DIR, exist_ok=True)
    os.makedirs(C.SOVITS_DIR, exist_ok=True)
    M_dict = {}

    logging.info("Start Server")
    app.run(host="0.0.0.0", port=6006)
    # 一个模型2.8GB, 3090一共24GB
    # gunicorn -w 4 -b 0.0.0.0:6006 GPT_SoVITS.GSV_server:app

    # 给队列发送结束信号
    for k, v in M_dict.items():
        p_list = v['process_list']
        for _ in range(len(p_list)):
            v['q_inp'].put(None)
        for p in p_list:
            p.join()

