import re
import numpy as np
import os
import logging
import base64
import json
import multiprocessing as mp
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from funasr import AutoModel

app = Flask(__name__, static_folder="./static_folder", static_url_path="")

# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  )


@app.route("/asr", methods=['POST'])
def asr_queue():
    global model
    info = request.get_json()
    inp_q, out_q = info["inp_queue"], info["out_queue"]
    mp.Value('')

    res = model.generate(input=inf["audio_buffer"],
                         batch_size_s=300,
                         hotword='魔搭')
    print(res)
    # return json.dumps({"code": 0, "msg": "", "result": res[0]['text']})
    return json.dumps({"code": 0, "msg": "", "result": res[0]})






