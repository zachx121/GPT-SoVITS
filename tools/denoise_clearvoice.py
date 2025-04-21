# 这个文件放到 /root/autodl-fs/GSV/ClearVoice/ClearerVoice-Studio/clearvoice/denoise_clearvoice.py
import sys
import os
import subprocess
from glob import glob
from tqdm.auto import tqdm
import logging
import torch
from faster_whisper import WhisperModel
from clearvoice import ClearVoice
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='[%(asctime)s-%(levelname)s-%(name)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

PROJ_DIR = "/root/GPT-SoVITS"
# python denoise_clearvoice_dev.py /root/autodl-fs/voice_sample/ChatTTS_Voice_Clone_User_3870_20250421164911734_aiqs/sliced /root/autodl-fs/voice_sample/ChatTTS_Voice_Clone_User_3870_20250421164911734_aiqs/denoised
if __name__ == '__main__':
    inp_dir = sys.argv[1]  # inp_dir="/root/autodl-fs/voice_sample/ChatTTS_Voice_Clone_User_3870_20250421164911734_aiqs/sliced"
    opt_dir = sys.argv[2]  # opt_dir="/root/autodl-fs/voice_sample/ChatTTS_Voice_Clone_User_3870_20250421164911734_aiqs/denoised_dev"
    logging.info(f"Denoise inp: {inp_dir}")
    logging.info(f"Denoise opt: {opt_dir}")
    myClearVoice = ClearVoice(task='speech_separation',
                              model_names=['MossFormer2_SS_16K'])
    # 处理目录 | online_write 处理一个音频写入一个音频
    logging.info(">>> ClearVoice降噪中...")
    myClearVoice(input_path=inp_dir,
                 output_path=opt_dir,
                 online_write=True)

    # 对s1、s2每个音频都调用ASR，哪个得分更高就用哪个，避免有时候s1、s2出现噪音、人声反过来的情况
    logging.info(">>> 加载ASR模型兜底人声、噪音的s1、s2变量错位的问题...")
    dir_whisper = os.path.join(PROJ_DIR, 'tools/asr/models/faster-whisper-large-v3-local')
    assert os.path.exists(dir_whisper)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 直接下载的目录还无法加载，需要根据配置指定到哪一个blobl
    with open(os.path.join(dir_whisper, "refs", "main"), "r") as fr:
        _version = fr.readlines()[0]
        fp_whisper = os.path.join(dir_whisper, "snapshots", _version)
        assert os.path.exists(fp_whisper)
    model = WhisperModel(fp_whisper, device=device, compute_type="float32")
    # 对每一个denoise的s1、s2使用whisper进行文本推理
    denoise_dir = os.path.join(opt_dir, "MossFormer2_SS_16K")
    wav_fp_list = glob(os.path.join(denoise_dir, "*s1.wav"))
    res_fp_list = []
    for s1_fp in wav_fp_list:
        s2_fp = s1_fp.replace("s1.wav", "s2.wav")
        assert os.path.exists(s1_fp)
        assert os.path.exists(s2_fp)
        segments, info = model.transcribe(
            audio=s1_fp,
            beam_size=5,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=700),
            language=None)
        s1_lang_prob = info.language_probability
        segments1 = list(segments)
        # 这个list里所有的avg_logprob是一样的
        s1_avg_logprob = segments1[0].avg_logprob if len(segments1) >= 1 else -999
        s1_nospeech_prob = segments1[0].no_speech_prob if len(segments1) >= 1 else -999

        segments, info = model.transcribe(
            audio=s2_fp,
            beam_size=5,
            # vad_filter=True,
            # vad_parameters=dict(min_silence_duration_ms=700),
            language=None)
        s2_lang_prob = info.language_probability
        segments2 = list(segments)
        # 这个list里所有的avg_logprob是一样的
        s2_avg_logprob = segments2[0].avg_logprob if len(segments2) >= 1 else -999
        s2_nospeech_prob = segments2[0].no_speech_prob if len(segments2) >= 1 else -999

        if s2_avg_logprob - s1_avg_logprob > 0.025:
            logging.info(f"\n>>> detect unexpected denoise result (gap: {s2_avg_logprob - s1_avg_logprob:.4f}), use s2 {s2_fp})")
            logging.info(f"s1_lang_prob:{s1_lang_prob}, s1_avg_logprob:{s1_avg_logprob}, s1_nospeech_prob:{s1_nospeech_prob} \nseg1:{segments1}")
            logging.info(f"s2_lang_prob:{s2_lang_prob}, s2_avg_logprob:{s2_avg_logprob}, s2_nospeech_prob:{s2_nospeech_prob} \nseg2:{segments2}")
            res_fp_list.append(s2_fp)
        else:
            res_fp_list.append(s1_fp)

    logging.info(">>> 将ASR分数更高的文件mv到目标目录")
    for fn in tqdm(res_fp_list):
        res = subprocess.run(["mv", fn, opt_dir],
                             capture_output=True, text=True,
                             encoding='utf-8')
        assert res.returncode == 0

    logging.info(">>> 清理临时目录")
    res = subprocess.run(["rm", "-rf", os.path.join(opt_dir, "MossFormer2_SS_16K")],
                         capture_output=True, text=True,
                         encoding='utf-8')
    assert res.returncode == 0

    # 处理混合语音文件
    # input_path = '/root/autodl-fs/audio_samples/带均匀底噪的训练音频_20250319.wav'
    # opt_path = '/root/autodl-fs/audio_samples/带均匀底噪的训练音频_20250319_denoised.wav'
    # output_wav = myClearVoice(input_path=input_path, online_write=False)
    # # output_wav[0]是声音，[1]是底噪
    # myClearVoice.write(output_wav, output_path=opt_path)



