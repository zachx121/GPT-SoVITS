
import torch
import numpy as np
import os
import logging
import base64
import json
from subprocess import getstatusoutput
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from GSV_model import GSVModel,ReferenceInfo

VOICE_SAMPLE_DIR = "/Users/bytedance/AudioProject/voice_sample/"
D_REF_SUFFIX = "default"
GPT_DIR = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/"
SOVITS_DIR = "/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/"


class Param:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. cn/en/fr/es)
    use_ref: bool = True  # 推理时是否使用参考音频的情绪
    ref_suffix: str = D_REF_SUFFIX  # 当可提供多个参考音频时，指定参考音频的后缀

    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        # {"JP": "all_ja", "ZH": "all_zh", "EN": "en", "ZH_EN": "zh", "JP_EN": "ja", "AUTO": "auto"}
        lang = self.lang if self.lang in ["JP", "ZH", "EN", "ZH_EN", "JP_EN"] else "AUTO"
        return lang

    @property
    def ref_info(self):
        ref_dir = os.path.join(VOICE_SAMPLE_DIR, self.speaker)
        # ~/tmp/ref_audio.wav
        audio_fp = os.path.join(ref_dir, f'ref_audio_{self.ref_suffix}.wav')
        # ~/tmp/ref_text.txt 里面必须是竖线分割的<语言>|<文本> e.g."ZH|语音的具体文本内容。"
        _text = open(os.path.join(ref_dir, f'ref_text_{self.ref_suffix}.wav')).readlines()[0].strip()
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
    app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")
    M: GSVModel = None
    # M = GSVModel(sovits_model_fp="/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/XiaoLinShuo_e4_s60.pth",
    #              gpt_model_fp="/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/XiaoLinShuo-e15.ckpt")

    @app.route("/init_model", methods=['POST'])
    def init_model():
        """
        http params:
        - speaker:str
        """
        global M
        if request.method != "POST":
            return "Only Support Post", 400
        info = request.get_json()
        M = GSVModel(sovits_model_fp=os.path.join(SOVITS_DIR, info['speaker']),
                     gpt_model_fp=os.path.join(GPT_DIR, info['speaker']),
                     speaker=info['speaker'])
        return "Init Success", 200

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
        os.makedirs(os.path.join(VOICE_SAMPLE_DIR, info['speaker']), exist_ok=True)
        suffix = info.get("ref_suffix", D_REF_SUFFIX)
        audio_fp = os.path.join(VOICE_SAMPLE_DIR, info['speaker'], f'ref_audio_{suffix}.wav')
        text_fp = os.path.join(VOICE_SAMPLE_DIR, info['speaker'], f'ref_text_{suffix}.txt')

        cmd = f"wget {info['ref_audio_url']} -O {audio_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            return f"Reference Audio Download Wrong", 500

        cmd = f"wget {info['ref_text_url']} -O {text_fp}"
        logging.debug(f"will execute `{cmd}`")
        status, output = getstatusoutput(cmd)
        if status != 0:
            return f"Reference Text Download Wrong", 500

    @app.route("/inference", methods=['POST'])
    def inference():
        """
        http params:
        - 以Param类的成员变量为准
        """
        global M
        if request.method != "POST":
            return "Only Support Post", 400
        if M is None:
            return "No available Model, request `init_model` first.", 400

        p = Param(request.get_json())
        if p.speaker != M.speaker:
            return f"inference使用的角色音({p.speaker})和当前初始化的模型角色音({M.speaker})不符", 400

        wav_sr, wav_arr = M.predict(target_text=p.text,
                                    target_lang=p.tgt_lang,
                                    ref_info=p.ref_info,
                                    top_k=1, top_p=0, temperature=0,
                                    ref_free=p.ref_free, no_cut=True)
        wav_arr_int16 = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
        rsp = {"trace_id": p.trace_id,
               "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
               "audio_buffer_int16": base64.b64encode(wav_arr_int16.tobytes()).decode(),
               "sample_rate": wav_sr,
               "status": "0",
               "msg": "success."}
        rsp = json.dumps(rsp)
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
        speaker = info['speaker']
        lang = info['lang']
        data_urls = info['data_urls']
        data_dir = os.path.join(VOICE_SAMPLE_DIR, speaker)
        os.makedirs(data_dir)
        logging.info(f">>> Start Data Preparing.")
        for url in data_urls:
            logging.info(f">>> Downloading Sample from {url}")
            status, output = getstatusoutput(f"wget {url} -P {data_dir}")
            if status != 0:
                logging.error(f"    Download fail. url is {url}")

        logging.info(f">>> Start Model Training.")
        cmd = f"python GPT_SoVITS/GSV_train.py {lang} {speaker} {data_dir}"
        logging.info(f"    cmd is {cmd}")
        status, output = getstatusoutput(cmd)
        if status != 0:
            logging.error("    Model training failed.")
            return "Model Training failed.", 500

    app.run(host="0.0.0.0", port=6006)


