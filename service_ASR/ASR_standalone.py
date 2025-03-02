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
import soundfile
logging.getLogger().setLevel(logging.DEBUG)
from faster_whisper import WhisperModel
import torch

# PROJ_DIR = "/Users/zhou/0-Codes/GPT-SoVITS"
PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))
print(f"PROJ_DIR: {PROJ_DIR}")
INP_QUEUE = "queue_self_asr_request"


# python servife_ASR.ASR_standalone audio_test.mp3
if __name__ == '__main__':
    audio_fp = sys.argv[1]
    audio_lang = sys.argv[2] if len(sys.argv) >= 3 else None
    dir_whisper = os.path.join(PROJ_DIR, 'tools/asr/models/faster-whisper-large-v3-local')
    assert os.path.exists(dir_whisper)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 直接下载的目录还无法加载，需要根据配置指定到哪一个blobl
    with open(os.path.join(dir_whisper, "refs", "main"), "r") as fr:
        _version = fr.readlines()[0]
        fp_whisper = os.path.join(dir_whisper, "snapshots", _version)
        assert os.path.exists(fp_whisper)
    model = WhisperModel(fp_whisper, device=device, compute_type="float32")

    segments, info = model.transcribe(
        audio=audio_fp,
        beam_size=5,
        # vad_filter=True,
        # vad_parameters=dict(min_silence_duration_ms=700),
        language=audio_lang)
    # text = "".join([seg.text for seg in segments])
    asr_fp = os.path.splitext(audio_fp)[0]+".txt"
    with open(asr_fp, "w") as fw:
        fw.writelines("\n".join([seg.text for seg in segments]))


