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

import torch
import time
import numpy as np
import os
import logging
import base64
import json
import shutil
from urllib.parse import unquote
from subprocess import getstatusoutput
from flask import Flask, request
import soundfile as sf
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
from GSV_model import GSVModel,ReferenceInfo

app = Flask(__name__, static_folder="./static_folder", static_url_path="")
D_REF_SUFFIX = "default"
VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("./voice_sample/"))
GPT_DIR = os.path.abspath(os.path.expanduser("./GPT_weights/"))
SOVITS_DIR = os.path.abspath(os.path.expanduser("./SoVITS_weights/"))
logging.info(f"VOICE_SAMPLE_DIR: {VOICE_SAMPLE_DIR}")
logging.info(f"GPT_DIR: {GPT_DIR}")
logging.info(f"SOVITS_DIR: {SOVITS_DIR}")

class Param:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. zh_cn/en_us)
    use_ref: bool = True  # 推理时是否使用参考音频的情绪
    ref_suffix: str = D_REF_SUFFIX  # 当可提供多个参考音频时，指定参考音频的后缀
    nocut: bool = True  # 是否不做切分

    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        # {"JP": "all_ja", "ZH": "all_zh", "EN": "en", "ZH_EN": "zh", "JP_EN": "ja", "AUTO": "auto"}
        # lang = self.lang if self.lang in ["JP", "ZH", "EN", "ZH_EN", "JP_EN"] else "AUTO"
        _lang_map = {"zh_cn": "ZH", "en_us": "EN"}
        lang = _lang_map.get(self.lang, "AUTO")
        return lang

    @property
    def ref_info(self):
        ref_dir = os.path.join(VOICE_SAMPLE_DIR, self.speaker)
        # ~/tmp/ref_audio.wav
        audio_fp = os.path.join(ref_dir, f'ref_audio_{self.ref_suffix}.wav')
        # ~/tmp/ref_text.txt 里面必须是竖线分割的<语言>|<文本> e.g."ZH|语音的具体文本内容。"
        _text = open(os.path.join(ref_dir, f'ref_text_{self.ref_suffix}.txt')).readlines()[0].strip()
        lang, text = _text.split("|")
        return ReferenceInfo(audio_fp=audio_fp, text=text, lang=lang)

    @property
    def ref_free(self):
        return False if self.use_ref else True

    def __init__(self, info_dict):
        for key in self.__annotations__.keys():
            if key in info_dict:
                setattr(self, key, info_dict[key])


if __name__ == '__main__':
    os.makedirs(VOICE_SAMPLE_DIR, exist_ok=True)
    os.makedirs(GPT_DIR, exist_ok=True)
    os.makedirs(SOVITS_DIR, exist_ok=True)
    M_dict: dict[str:GSVModel] = {}
    # M: GSVModel = None
    # M = GSVModel(sovits_model_fp="~/AudioProject/GPT-SoVITS/SoVITS_weights/XiaoLinShuo_e4_s60.pth",
    #              gpt_model_fp="~/AudioProject/GPT-SoVITS/GPT_weights/XiaoLinShuo-e15.ckpt")

    # @app.route("/init_status", methods=['POST'])
    # def init_status():
    #     """
    #     """
    #     global M
    #     if request.method != "POST":
    #         return "Only Support Post", 400
    #     if M is None:
    #         return json.dumps({"is_init": 0}), 200
    #     else:
    #         return json.dumps({"is_init": 1}), 200


    @app.route("/init_model", methods=['POST'])
    def init_model():
        """
        http params:
        - speaker:str
        """
        res = {"status": 0, "msg": "", "result": ""}
        info = request.get_json()
        logging.debug(info)
        if info["speaker"] in M_dict:
            res['status'] = 1
            res['msg'] = "Already Inited"
            return json.dumps(res)

        import torch
        device = torch.device("cuda")
        mem_allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
        mem_cached = torch.cuda.memory_cached(device) / 1024 ** 2
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        used_mem = mem_allocated + mem_cached
        free_mem = total_mem - used_mem
        logging.debug(f"Total memory: {total_mem} MB, Used memory: {used_mem} MB, Free memory: {free_mem} MB")
        if free_mem <= 2800*2:
            res['status'] = 1
            res['msg'] = 'GPU OOM'
            return json.dumps(res)
        M = GSVModel(sovits_model_fp=os.path.join(SOVITS_DIR, info['speaker'], info['speaker']+".latest.pth"),
                     gpt_model_fp=os.path.join(GPT_DIR, info['speaker'], info['speaker']+".latest.ckpt"),
                     speaker=info['speaker'])
        M_dict[info['speaker']] = M
        res['msg'] = "Init Success"
        return json.dumps(res)


    @app.route("/unload_model", methods=['POST'])
    def unload_model():
        info = request.get_json()
        logging.debug(info)
        del M_dict[info["speaker"]]
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        res = {"status": 0,
               "msg": "Model Unloaded",
               "result": ""}
        return json.dumps(res)

    @app.route("/add_reference", methods=['POST'])
    def add_reference():
        """
        http params:
        - speaker:str
        - ref_audio_url:str
        - ref_text_url:str
        - ref_suffix:str optional 当可以提供多个参考音频时，可以用情绪作为后缀
        """
        if request.method != "POST":
            return "Only Support Post", 400
        info = request.get_json()
        logging.debug(info)
        os.makedirs(os.path.join(VOICE_SAMPLE_DIR, info['speaker']), exist_ok=True)
        suffix = info.get("ref_suffix", D_REF_SUFFIX)
        audio_fp = os.path.join(VOICE_SAMPLE_DIR, info['speaker'], f'ref_audio_{suffix}.wav')
        text_fp = os.path.join(VOICE_SAMPLE_DIR, info['speaker'], f'ref_text_{suffix}.txt')

        cmd = f"wget \"{info['ref_audio_url']}\" -O {audio_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"status": 0,
                              "msg": f"Reference Audio Download Wrong",
                              "result": ""})
            return res

        cmd = f"wget \"{info['ref_text_url']}\" -O {text_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            res = json.dumps({"status": 0,
                              "msg": f"Reference Text Download Wrong",
                              "result": ""})
            return res

        res = json.dumps({"status": 0,
                          "msg": "Reference added.",
                          "result": ""})
        return res

    @app.route("/inference", methods=['POST'])
    def inference():
        """
        http params:
        - 以Param类的成员变量为准
        """
        if request.method != "POST":
            return "Only Support Post", 400
        if len(M_dict) == 0:
            return "No available Model", 400

        info = request.get_json()
        logging.debug(info)
        p = Param(info)
        if p.speaker not in M_dict:
            return f"inference使用的角色音({p.speaker})未被加载。已加载角色音: {M_dict.keys()}", 400
        M = M_dict[p.speaker]
        wav_sr, wav_arr = M.predict(target_text=p.text,
                                    target_lang=p.tgt_lang,
                                    ref_info=p.ref_info,
                                    top_k=1, top_p=0, temperature=0,
                                    ref_free=p.ref_free, no_cut=p.nocut)
        wav_arr_int16 = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
        rsp = {"trace_id": p.trace_id,
               # "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
               "audio_buffer_int16": base64.b64encode(wav_arr_int16.tobytes()).decode(),
               "sample_rate": wav_sr,
               "status": 0,
               "msg": "success."}
        rsp = json.dumps({"status": 0,
                          "msg": "",
                          "result": rsp})
        sf.write(f"output_{time.time():.0f}.wav", wav_arr, wav_sr)
        return rsp, 200

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
        speaker = info['speaker']
        lang = info['lang']
        data_urls = info['data_urls']
        data_dir = os.path.join(VOICE_SAMPLE_DIR, speaker)
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
            return "All Audio url failed to download.", 400
        logging.info(f">>> Start Model Training.")
        # python GPT_SoVITS/GSV_train.py zh_cn test_silang1636 voice_sample/test_silang1636 > test_silang1636.train
        cmd = f"nohup python GPT_SoVITS/GSV_train.py {lang} {speaker} {data_dir} > {speaker}.train 2>&1 &"
        logging.info(f"    cmd is {cmd}")
        status, output = getstatusoutput(cmd)
        if status != 0:
            logging.error("    Model training failed.")
            res = json.dumps({"status": 0,
                              "msg": "Model Training Failed",
                              "result": ""})
            st_code = 500
        else:
            res = json.dumps({"status": 0,
                              "msg": "Model Training Started.",
                              "result": ""})
            st_code = 500
        return res, st_code

    @app.route("/model_status", methods=['POST'])
    def model_status():
        all_loaded_speakers = list(M_dict.keys())
        logging.debug(", ".join(all_loaded_speakers))
        res = json.dumps({"status": 0,
                          "msg": "",
                          "result": all_loaded_speakers})
        return res, 200

    @DeprecationWarning
    @app.route("/is_model_available", methods=['POST'])
    def is_model_available():
        """
        模型训练是否完成
        http params:
        - speaker_list: list[str]
        """
        info = request.get_json()
        speaker_list = info['speaker_list']
        status = {}
        for s in speaker_list:
            cond1 = os.path.exists(os.path.join(SOVITS_DIR, s, s+".latest.pth"))
            cond2 = os.path.exists(os.path.join(GPT_DIR, s, s+".latest.ckpt"))
            status[s] = cond1 and cond2
        return json.dumps(status), 200

    app.run(host="0.0.0.0", port=6006)
    # 一个模型2.8GB, 3090一共24GB
    # gunicorn -w 4 -b 0.0.0.0:6006 GPT_SoVITS.GSV_server:app
