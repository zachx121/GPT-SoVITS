import sys

import requests
import json
import numpy as np
import base64
import scipy
import time
from tqdm.auto import tqdm
import multiprocessing as mp
import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s-%(funcName)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

url = "https://u212392-8449-c474cb97.beijinga.seetacloud.com/"

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def train(sid, data_urls, lang):
    assert lang in {"zh_cn", "en_us", "auto"}
    logging.info(">>> start train")
    rsp = requests.post(url + "train_model",
                        data=json.dumps({"speaker": sid,
                                         "lang": lang,
                                         "data_urls": data_urls}),
                        headers=headers)
    print(rsp.status_code, rsp.json())
    for i in range(50):
        time.sleep(5)
        rsp = requests.post(url + "check_training_status", data=json.dumps({}), headers=headers)
        print(rsp.status_code, rsp.json())

def load(sid):
    logging.info(">>> start load")
    rsp = requests.post(url + "load_model",
                        data=json.dumps({"speaker": sid, "speaker_num": 2}),
                        headers=headers,
                        timeout=10)
    print(rsp.status_code, rsp.json())

def unload(sid):
    logging.info(">>> start unload")
    rsp = requests.post(url + "unload_model",
                        data=json.dumps({"speaker": sid}),
                        headers=headers)
    print(rsp.status_code, rsp.json())

def add_ref(sid):
    logging.info(">>> start add_ref")
    # 只传一个sid，表示自动用训练集的第一个音频作为reference
    rsp = requests.post(url + "add_reference",
                        data=json.dumps({"speaker": sid}),
                        headers=headers)
    print(rsp.status_code, rsp.json())

def is_model_oss_available(sid):
    rsp = requests.post(url + "is_model_oss_available",
                        data=json.dumps({"speaker_list": [sid]}),
                        headers=headers)
    print(rsp.status_code, rsp.json())
    return rsp.json()['result'][0]['is_available']

def model_status(sid):
    logging.info(">>> start model_status")
    rsp = requests.post(url + "model_status", headers=headers)
    print(rsp.status_code, rsp.json())

def inference(sid,
              lang="en_us",
              text="Hello, I'm not sure I understand your description",
              trace_id="debug_GSV_client"):
    logging.info(">>> start inference")
    # # 注意 lang 和 text一定要对齐，不能用text是中文文本然后lang又指定英文，100%泄漏reference音频或者乱读
    rsp = requests.post(url + "inference",
                        data=json.dumps({"trace_id": trace_id,
                                         "speaker": sid,
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
sid, lang, sid_data_urls = "fuhang_1", "zh_cn", ["https://public.yisounda.com/fuhang.m4a?e=1729698462&token=izz8Pq4VzTJbD8CmM3df5BAncyqynkPgF1K4srqP:c6qaV6h2qIyUX03u4gJQILb8Ipo="]
sid, lang, sid_data_urls = "ChatTTS_Voice_Clone_0_Mike_komd", "en_us", ["http://resource.aisounda.cn/model%2Fclone%2Fself%2F97ba7a9c-7602-4cdb-994d-0f286fa4b99d.m4a?e=1730046614&token=izz8Pq4VzTJbD8CmM3df5BAncyqynkPgF1K4srqP:mOgJ45UzLaRZiJbN0cA9CQLZMTs="]
# train(sid, sid_data_urls, lang=lang)
assert is_model_oss_available(sid)
# sys.exit(0)
load(sid)
model_status(sid)
# sys.exit(0)
sta = time.time()
# add_ref(sid)  # 不需要了，load会自动指定默认ref
# inference(sid, text=f"this is a general sentence of number", trace_id=f"debug_")
inference(sid, lang="zh_cn", text=f"你好啊，我的朋友。很高兴认识你！你们在做什么事情呢？", trace_id=f"debug_")
inference(sid, lang="en_us", text=f"hi there. how is your day? I'm glad to meet you", trace_id=f"debug_")
end = time.time()
print(f"duration: {end-sta:.02f}s")
sys.exit(0)
# 多进程并发请求
def func(i):
    sta = int(time.time() * 1000)
    logging.info(f"Inference of idx={i}")
    # inference(sid, text=f"this is a general sentence of number {i}", trace_id=f"debug_{i}")
    inference(sid, lang="zh_cn", text=f"你好啊，我的朋友。很高兴认识你！你们在做什么事情呢？", trace_id=f"debug_{i}")
    end = int(time.time() * 1000)
    return i, {"sta": sta, "end": end, "duration": end - sta}
mp.set_start_method("fork")
with mp.Pool(20) as p:
    all_res = p.map(func, range(20))
    info = dict(all_res)
print("time as follow: ")
print(info)


def standalone_debug():
    # 在上层目录执行
    import os
    import sys
    sys.path.append("/root/GPT-SoVITS/GPT_SoVITS")
    from GSV_model import GSVModel, ReferenceInfo
    from GSV_const import InferenceParam
    import numpy as np
    import librosa
    import scipy

    VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("./voice_sample/"))
    GPT_DIR = os.path.abspath(os.path.expanduser("./GPT_weights/"))
    SOVITS_DIR = os.path.abspath(os.path.expanduser("./SoVITS_weights/"))

    sid = "fuhang"
    M = GSVModel(sovits_model_fp=os.path.join(SOVITS_DIR, sid, sid + ".latest.pth"),
                 gpt_model_fp=os.path.join(GPT_DIR, sid, sid + ".latest.ckpt"),
                 speaker=sid)
    p = InferenceParam({"trace_id": "dd", "speaker": sid, "text": "测试音频效果", "lang": "zh_cn"})
    p = InferenceParam({"trace_id": "dd", "speaker": sid, "text": "I don't understand your meanings", "lang": "en_us"})

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


