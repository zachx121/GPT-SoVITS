
import os
import numpy as np
from funasr import AutoModel
from .GSV_const import Route as R
from .GSV_const import ReferenceInfo
from .GSV_model import GSVModel
import wave
import torch
import io
import logging
import scipy
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from flask import Flask, request, jsonify
import base64
import json

# PROJ_DIR = "/root/GPT-SoVITS"
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))


class Model:
    def __init__(self, tts_sid):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TTS
        logging.info(">>> Init TTS...")
        self.sid = tts_sid
        sovits_fp = R.get_sovits_fp(self.sid)
        gpt_fp = R.get_gpt_fp(self.sid)
        self.tts_model = GSVModel(sovits_model_fp=sovits_fp, gpt_model_fp=gpt_fp)

        # ASR
        logging.info(">>> Init ASR...")
        path_asr = os.path.join(PROJ_DIR, 'tools/asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
        path_vad = os.path.join(PROJ_DIR, 'tools/asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch')
        path_punc = os.path.join(PROJ_DIR, 'tools/asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
        for fp in [path_asr, path_vad, path_punc]:
            assert os.path.exists(fp), f"ASR Model Not Exist: path='{fp}'"
        self.asr_model = AutoModel(
            model=path_asr,
            model_revision="v2.0.4",
            vad_model=path_vad,
            vad_model_revision="v2.0.4",
            punc_model=path_punc,
            punc_model_revision="v2.0.4",
        )

        # Translation
        logging.info(">>> Init Translation...")
        cache_dir = os.path.join(PROJ_DIR, "tools/translate")
        self.translate_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir, local_files_only=True)
        self.translate_model.to(self.device)  # Move model to GPU
        self.translate_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=cache_dir, local_files_only=True)

    def asr(self, audio):
        assert audio.dtype == np.float32
        res = self.asr_model.generate(input=audio, disable_pbar=True)
        return res[0]['text']

    def trans(self, text, src_lang="zh", tgt_lang="en"):
        # zh/ja/en/ko 共计有100个语言
        assert src_lang in self.translate_tokenizer.lang_code_to_id.keys(), f"src_lang='{src_lang}', not support"
        assert tgt_lang in self.translate_tokenizer.lang_code_to_id.keys(), f"tgt_lang='{tgt_lang}', not support"
        # chinese_text = "生活就像一盒巧克力。"
        self.translate_tokenizer.src_lang = src_lang
        tgt_lang_id = self.translate_tokenizer.get_lang_id(tgt_lang)
        encoded_text = self.translate_tokenizer(text, return_tensors="pt")
        encoded_text = {k: v.to(self.device) for k, v in encoded_text.items()}
        generated_tokens = self.translate_model.generate(**encoded_text,
                                                         forced_bos_token_id=tgt_lang_id)
        return self.translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    def tts(self, line, tgt_lang, ref_info: ReferenceInfo = None):
        """
        Args:
            line: 文本
            tgt_lang: 只要在LANG_MAP里就行，key还是value都可以，predict内部会自行映射到en/all_zh/all_jp/all_ko
            ref_info:
        Returns:

        """
        ref_info = ref_info if ref_info is not None else ReferenceInfo.from_sid(self.sid)
        sr, audio, tts_num = self.tts_model.predict(target_text=line,
                                                    target_lang=tgt_lang,
                                                    ref_info=ref_info,
                                                    top_k=30, top_p=0.99, temperature=0.6,
                                                    no_cut=True)
        audio = Model.resample_audio(audio, sr, 16000)
        sr = 16000
        return sr, audio

    def predict(self, audio, src_lang, tgt_lang):
        asr_txt = self.asr(audio)
        trans_txt = self.trans(asr_txt, src_lang, tgt_lang)
        sr, audio = self.tts(trans_txt, tgt_lang)
        return sr, audio, asr_txt, trans_txt

    @staticmethod
    def resample_audio(audio_arr_int16, original_rate=32000, target_rate=16000):
        # 计算重采样后的长度
        resampled_length = int(len(audio_arr_int16) * (target_rate / original_rate))
        # 进行重采样
        resampled_audio = scipy.signal.resample(audio_arr_int16, resampled_length)
        # 将重采样后的音频数据转换回 int16 类型
        resampled_audio_int16 = np.int16(resampled_audio)
        return resampled_audio_int16


def dev():
    M = Model("amber")
    import librosa
    import time
    fp = "/root/autodl-fs/audio_samples/董宇辉带货.m4a"
    # Audio(fp)
    audio_arr, sr = librosa.load(fp, sr=16000, mono=True)

    b = time.time()
    txt = M.asr(audio_arr)
    e_asr = time.time()
    txt_trans = M.trans(txt, "zh", "en")
    e_trans = time.time()
    sr, audio = M.tts(txt_trans, "en")
    e_tts = time.time()
    print(f"[ASR]: {(e_asr - b) * 1000:.0f}ms")
    print(f"[Trans]: {(e_trans - e_asr) * 1000:.0f}ms")
    print(f"[TTS]: {(e_tts - e_trans) * 1000:.0f}ms")
    print(f"[Total]: {(e_tts - b) * 1000:.0f}ms")
    print(f"txt: {txt}\ntxt_trans: {txt_trans}\nelapsed: {(e_tts - b) * 1000:.0f}ms")
    # Audio(audio, rate=sr)


M = Model("lydia")
app = Flask(__name__)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    info = request.get_json()
    audio_buffer = info['audio_buffer_16khz_int16']
    audio_buffer = base64.b64decode(audio_buffer)
    src_lang = info['source_lang']
    tgt_lang = info['target_lang']
    audio_arr_int16_16khz = np.frombuffer(audio_buffer, dtype=np.int16)
    audio_arr_float32_16khz = audio_arr_int16_16khz.astype(np.float32) / 32768.0
    sr, audio_arr_int16_16khz, asr_txt, trans_txt = M.predict(audio_arr_float32_16khz, src_lang, tgt_lang)
    rsp = {"audio_buffer_int16": base64.b64encode(audio_arr_int16_16khz.tobytes()).decode(),
           "sample_rate": sr,  # 32khz
           "asr_txt": asr_txt,
           "trans_txt": trans_txt,
           }
    print(f"推理结果 asr_txt='{asr_txt}'")
    print(f"推理结果 trans_txt='{trans_txt}'")
    rsp = json.dumps({"code": 0,
                      "msg": "",
                      "result": rsp})
    return rsp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8002)

