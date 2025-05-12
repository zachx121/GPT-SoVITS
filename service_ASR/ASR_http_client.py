import librosa
import requests
import base64
import json
import numpy as np
import time
import multiprocessing as mp

url = "https://u212392-ae2e-51fb6d07.bjb1.seetacloud.com:8443/transcribe"
fp = "/Users/zhou/0-Codes/VoiceSamples/0-角色音样本/董宇辉带货.mp3"


def send_request(trace_id_suffix):
    y, sr = librosa.load(fp, mono=True, sr=16000)
    y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    stime = time.time()
    payload = {"trace_id": f"{int(stime * 1000)}_{trace_id_suffix}",
               "audio_buffer": base64.b64encode(y_int16.tobytes()).decode()}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    etime = time.time()
    print(f"进程 {payload['trace_id']}: {response}")
    print(f"进程 {payload['trace_id']} 内容: {response.text[:100]}")
    print(f"进程 {payload['trace_id']} 耗时: {(etime - stime) * 1000:.0f}ms")


if __name__ == '__main__':
    processes = []
    for i in range(5):  # 创建10个进程
        p = mp.Process(target=send_request, args=(i,))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

