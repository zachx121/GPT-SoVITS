
import torch
import numpy as np
import logging
import base64
import json
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from GSV_model import GSVModel


class Param:
    trace_id: str = ""
    buffer: bytes = None  # 音频按base64编码的字节流，会使用base64.b64decode(p.buffer)解码
    buffer_dtype: str = "int16"
    sample_rate: float = 16000
    speed: float = 1.0  # 合成音频的语速 (e.g. 1.2加速 0.8减速)
    lang: str = None  # 合成音频的语言 (e.g. cn/en/fr/es)

    @property
    def dtype(self):
        dtype_map = {"int16": np.int16, "float32": np.float32}
        return dtype_map.get(self.buffer_dtype, None)

    # 模型接收的参数是“时长” 所以入参是"速度"就需要改一下
    @property
    def duration_factor(self):
        return round(1 / self.speed, 4)

    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        language_map = {"zh_cn": "cmn", "en_us": "eng", "fr_fr": "fra", "es_es": "spa"}
        assert self.lang in language_map, f"语言参数错误: lang='{self.lang}'，只接受 {'/'.join(language_map.keys())}"
        return language_map[self.lang]

    def __init__(self, info_dict):
        for key in self.__annotations__.keys():
            if key in info_dict:
                setattr(self, key, info_dict[key])

        # 暂时强制要求音频类型为16khz、int16，后端这里不做重采样
        assert self.dtype == np.int16 and self.sample_rate == 16000, "暂时强制要求音频类型为16khz、int16，后端这里不做重采样"



if __name__ == '__main__':
    app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")
    M = ExpressiveModel()


    # POST | 如果是json
    @app.route("/test_json", methods=['POST'])
    def test_json():
        if request.method != "POST":
            return None

        p = Param(request.get_json())

        audio_arr = np.frombuffer(base64.b64decode(p.buffer), dtype=p.dtype)

        # M.predict要求是双通道float32的音频，而输入是单通道int16
        if len(audio_arr.shape) == 1:
            # audio_arr = np.vstack([audio_arr, audio_arr])
            audio_arr = np.expand_dims(audio_arr, axis=0)
        audio_arr = audio_arr.astype(np.float32) / 32768.0  # 归一化到 [-1.0, 1.0]
        audio_arr = torch.from_numpy(audio_arr.T)
        wav_arr, wav_sr, text_cstr = M.predict(audio_arr,
                                               duration_factor=p.duration_factor,
                                               tgt_lang=p.tgt_lang)
        wav_arr_int16 = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
        rsp = {"trace_id": p.trace_id,
               "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
               "audio_buffer_int16": base64.b64encode(wav_arr_int16.tobytes()).decode(),
               "sample_rate": wav_sr,
               "audio_text": str(text_cstr),
               "status": "0",
               "msg": "success."}
        print(rsp['audio_text'])
        rsp = json.dumps(rsp)
        return rsp


    app.run(host="0.0.0.0", port=6006)


