import sys

import requests
import json
import numpy as np
import base64
import scipy
import time

url = "https://u212392-8449-c474cb97.beijinga.seetacloud.com/"

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def train(sid):
    rsp = requests.post(url + "train_model",
                        data=json.dumps({"speaker": sid,
                                         "lang": "zh_cn",
                                         "data_urls": ['https://public.yisounda.com/tmp.wav?e=1729227642&token=izz8Pq4VzTJbD8CmM3df5BAncyqynkPgF1K4srqP:sb8W5U6yEvfMd6wW7q2UC7wI-8w=']}),
                        headers=headers)
    print(rsp.status_code, rsp.json())
    pass

def load(sid):
    rsp = requests.post(url + "load_model",
                        data=json.dumps({"speaker": sid, "speaker_num": 2}),
                        headers=headers)
    print(rsp.status_code, rsp.json())

def unload(sid):
    rsp = requests.post(url + "unload_model",
                        data=json.dumps({"speaker": sid}),
                        headers=headers)
    print(rsp.status_code, rsp.json())

def add_ref(sid):
    # 只传一个sid，表示自动用训练集的第一个音频作为reference
    rsp = requests.post(url + "add_reference",
                        data=json.dumps({"speaker": sid}),
                        headers=headers)
    print(rsp.status_code, rsp.json())

def model_status(sid):
    rsp = requests.post(url + "model_status", headers=headers)
    print(rsp.status_code, rsp.json())

def inference(sid,
              lang="en_us",
              text="Hello, I'm not sure I understand your description"):
    # # 注意 lang 和 text一定要对齐，不能用text是中文文本然后lang又指定英文，100%泄漏reference音频或者乱读
    rsp = requests.post(url + "inference",
                        data=json.dumps({"trace_id": "debug_GSV_client", "speaker": sid,
                                         "text": text,
                                         "lang": lang,
                                         # "text": "你好，我正在尝试解答这个问题",
                                         #  "lang": "zh_cn",
                                         "use_ref": True}),
                        headers=headers)
    print(rsp.status_code)
    if rsp.status_code == 200:
        rsp_audio_arr = np.frombuffer(base64.b64decode(rsp.json()['result']['audio_buffer_int16']), dtype=np.int16)
        scipy.io.wavfile.write(f"./rsp_{time.time():.0f}.wav", 16000, rsp_audio_arr)


sid = 'test_silang1636'
# sid = 'ChatTTS_Voice_Clone_4_222rb2j'
train(sid)
for i in range(10):
    time.sleep(5)
    rsp = requests.post(url + "check_training_status", data=json.dumps({}), headers=headers)
    print(rsp.status_code, rsp.json())

# sys.exit(0)
load(sid)
model_status(sid)
inference(sid)

# rsp = requests.post(url + "is_model_available",
#                     data=json.dumps({"speaker_list": ["test_silang1636", "ChatTTS_Voice_Clone_4_222rb2j", "test_silang4"]}),
#                     headers=headers)
# print(rsp.status_code, rsp.json())


def standalone_debug():
    # 在上层目录执行
    import os
    import sys
    sys.path.append("/root/GPT-SoVITS/GPT_SoVITS")
    from GSV_model import GSVModel, ReferenceInfo
    from GSV_server import Param
    import numpy as np
    import librosa
    import scipy

    VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("./voice_sample/"))
    GPT_DIR = os.path.abspath(os.path.expanduser("./GPT_weights/"))
    SOVITS_DIR = os.path.abspath(os.path.expanduser("./SoVITS_weights/"))

    sid = "test_silang1636"
    M = GSVModel(sovits_model_fp=os.path.join(SOVITS_DIR, sid, sid + ".latest.pth"),
                 gpt_model_fp=os.path.join(GPT_DIR, sid, sid + ".latest.ckpt"),
                 speaker=sid)
    p = Param({"trace_id": "dd", "speaker": sid, "text": "测试音频效果", "lang": "zh_cn"})
    p = Param({"trace_id": "dd", "speaker": sid, "text": "I don't understand your meanings", "lang": "en_us"})

    wav_sr, wav_arr_int16 = M.predict(target_text=p.text,
                                      target_lang=p.tgt_lang,
                                      ref_info=p.ref_info,
                                      top_k=1, top_p=0, temperature=0,
                                      ref_free=p.ref_free, no_cut=p.nocut)
    print(wav_arr_int16.shape)
    wav_arr_float32 = wav_arr_int16.astype(np.float32)/32768.0
    wav_arr_float32_16khz = librosa.resample(wav_arr_float32, orig_sr=wav_sr, target_sr=16000)
    wav_arr_int16 = (np.clip(wav_arr_float32_16khz, -1.0, 1.0) * 32767).astype(np.int16)
    scipy.io.wavfile.write("/root/GPT-SoVITS/GPT_SoVITS/audio.wav", 16000, wav_arr_int16)


