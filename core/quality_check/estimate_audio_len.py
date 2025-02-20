import numpy as np
import subprocess
from service_GSV.GSV_model import GSVModel
from service_GSV.GSV_const import Route as R
from service_GSV.GSV_const import ReferenceInfo
from service_GSV import GSV_const as C
import soundfile as sf
import re
import os
import sys

SEP_MAP = {"zh_cn": ["。", "，", "？", "！", ",", "?", "!"],
           "en_us": [".", ",", "?", "!"]}

if __name__ == '__main__':

    lang = "zh_cn"
    lang = "en_us"
    all_sep = "".join(SEP_MAP[lang])
    text_list = []
    text_fp = os.path.abspath(os.path.join(__file__, "..", f"text_{lang}.txt"))
    with open(text_fp) as fr:
        for i in fr.readlines():
            for j in re.split(f"[{all_sep}]", i.strip("\n")):
                j = j.strip()
                if len(j) != 0:
                    text_list.append(j)
    print(text_list)
    sys.exit(0)

    sid = ""
    sovits_model = R.get_sovits_fp(sid)
    gpt_model = R.get_gpt_fp(sid)
    M = GSVModel(sovits_model_fp=sovits_model, gpt_model_fp=gpt_model)

    ref_audio_fp = R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)
    ref_lang, ref_text = subprocess.getoutput(f"cat {R.get_ref_text_fp(sid, C.D_REF_SUFFIX)}").strip().split("|")
    ref_info = ReferenceInfo(audio_fp=ref_audio_fp, lang=ref_lang, text=ref_text)

    for text in text_list:
        wav_sr, wav_arr_int16 = M.predict(target_text=text,
                                          target_lang=lang,
                                          ref_info=ref_info,
                                          top_k=30, top_p=0.99, temperature=0.3,
                                          ref_free=False, no_cut=False)
        num_tokens = text.split(" ") if lang == "en_us" else len(text)
        audio_duration = wav_arr_int16.shape[0]/wav_sr
