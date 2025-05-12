import re
import sys

import numpy as np
import wave
import os
import logging
import base64
import json
import pika
import librosa
import torchaudio
import time
import gzip
from logging.handlers import TimedRotatingFileHandler
import multiprocessing as mp
import scipy
import torch
import soundfile
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from funasr import AutoModel
from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
import numpy as np
import traceback
# PROJ_DIR = "/Users/zhou/0-Codes/GPT-SoVITS"
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))
print(f"PROJ_DIR: {PROJ_DIR}")

class ASRModelWrapper:
    def __init__(self, lang):
        assert lang in ["zh_cn", "en_us"]
        self.lang = lang
        self.model = self.load_zh_asr_model() if self.lang=="zh_cn" else self.load_other_asr_model()

    def predict(self, inp):
        # inp: audio_arr_float32 or file_path
        if self.lang == "zh_cn":
            # res: [{'key': '董宇辉带货_16k_mono', 'text': '那会儿人们对于沟通是有多么强烈的欲望呀，啊就像那些痛苦时刻的诗人有那么强烈的。'}]
            res = self.model.generate(input=inp, disable_pbar=True)
            if res is not None and len(res) > 0:
                asr_txt = res[0]['text']
            else:
                asr_txt = ""
            return asr_txt
        else:
            segments, info = self.model.transcribe(
                audio=inp,
                beam_size=5,
                # vad_filter=True,
                # vad_parameters=dict(min_silence_duration_ms=700),
                language=None)
            asr_txt = "".join([seg.text for seg in segments])
            return asr_txt

    @staticmethod
    def load_zh_asr_model():
        logger.info(f">>> loading asr model with [damo]")
        # paraformer-zh is a multi-functional asr model
        # use vad, punc, spk or not as you need
        path_asr = os.path.join(PROJ_DIR, 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
        path_vad = os.path.join(PROJ_DIR, 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch')
        path_punc = os.path.join(PROJ_DIR, 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
        for fp in [path_asr, path_vad, path_punc]:
            assert os.path.exists(fp), f"ASR Model Not Exist: path='{fp}'"
        """
        去掉VAD因为会报错：
        ERROR:__main__:list index out of range
        ERROR:__main__:Traceback (most recent call last):
          File "/root/GPT-SoVITS/service_ASR/ASR_http.py", line 118, in transcribe_audio
            asr_txt = model.predict(inp=audio_arr_float32)
          File "/root/GPT-SoVITS/service_ASR/ASR_http.py", line 41, in predict
            res = self.model.generate(input=inp, disable_pbar=True)
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/auto/auto_model.py", line 306, in generate
            return self.inference_with_vad(input, input_len=input_len, **cfg)
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/auto/auto_model.py", line 383, in inference_with_vad
            res = self.inference(
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/auto/auto_model.py", line 345, in inference
            res = model.inference(**batch, **kwargs)
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/models/fsmn_vad_streaming/model.py", line 722, in inference
            segments_i = self.forward(**batch)
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/models/fsmn_vad_streaming/model.py", line 566, in forward
            self.DetectLastFrames(cache=cache)
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/models/fsmn_vad_streaming/model.py", line 772, in DetectLastFrames
            frame_state = self.GetFrameState(
          File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/funasr/models/fsmn_vad_streaming/model.py", line 495, in GetFrameState
            cur_decibel = cache["stats"].decibel[t]
        IndexError: list index out of range
        """
        model = AutoModel(
            model=path_asr,
            model_revision="v2.0.4",
            # vad_model=path_vad,
            # vad_model_revision="v2.0.4",
            punc_model=path_punc,
            punc_model_revision="v2.0.4",
            disable_update=True
        )
        return model

    @staticmethod
    def load_other_asr_model():
        logger.info(f">>> loading asr model with [whisper]")
        # paraformer-zh is a multi-functional asr model
        # use vad, punc, spk or not as you need
        dir_whisper = os.path.join(PROJ_DIR, 'tools/asr/models/faster-whisper-large-v3-local')
        assert os.path.exists(dir_whisper)
        # 直接下载的目录还无法加载，需要根据配置指定到哪一个blobl
        with open(os.path.join(dir_whisper, "refs", "main"), "r") as fr:
            _version = fr.readlines()[0]
            fp_whisper = os.path.join(dir_whisper, "snapshots", _version)
            assert os.path.exists(fp_whisper)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = WhisperModel(fp_whisper, device='cuda', compute_type="float16")
        return model


# 假设已经全局初始化模型
model = ASRModelWrapper(lang="zh_cn")
# warm-up
for _ in range(5):
    res = model.predict(inp="/root/autodl-fs/audio_samples/董宇辉带货_16k_mono.wav")
    logger.debug(f"warm-up inference. asr result:{res}")

app = Flask(__name__)


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # 从请求中获取音频数据
        s0 = time.time()
        param = request.get_json()
        tid = param.get("trace_id", "")
        logger.debug(f"拿到数据耗时 {s0*1000 - int(tid.split('_')[0]):.0f}ms")
        logger.debug(f"解析数据耗时 {(time.time() - s0)*1000:.0f}ms")
        audio_bytes = base64.b64decode(param["audio_buffer"])
        audio_arr_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_arr_float32 = audio_arr_int16.astype(np.float32) / 32768.0
        s1 = time.time()
        asr_txt = model.predict(inp=audio_arr_float32)
        s2 = time.time()
        logger.debug(f"推理耗时{(s2-s1)*1000:.0f}ms, asr:{asr_txt}")
        rsp = {"trace_id": tid,
               "audio_text": asr_txt,
               "finish_time": int(time.time() * 1000)}
        rsp = {"code": 0,
               "msg": "",
               "result": rsp}
        rsp = json.dumps(rsp, ensure_ascii=False)  # ensure_ascii
        # 返回转录结果
        return rsp
    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return None


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6006
    app.run(host='0.0.0.0', port=port)

