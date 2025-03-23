# 这个文件放到 /root/autodl-fs/GSV/ClearVoice/ClearerVoice-Studio/clearvoice/denoise_clearvoice.py
import sys
import os
import subprocess
from glob import glob
# from tqdm import tqdm
import logging
from clearvoice import ClearVoice
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

if __name__ == '__main__':
    inp_dir = sys.argv[1]  # inp_dir="/root/autodl-fs/voice_sample/lydia/slicer_opt"
    opt_dir = sys.argv[2]  # opt_dir="/root/autodl-fs/voice_sample/lydia/denoise_opt_dev"
    myClearVoice = ClearVoice(task='speech_separation',
                              model_names=['MossFormer2_SS_16K'])
    # 处理目录 | online_write 处理一个音频写入一个音频
    myClearVoice(input_path=inp_dir,
                 output_path=opt_dir,
                 online_write=True)
    # 格式化
    reg_fp = os.path.join(opt_dir, "MossFormer2_SS_16K", "*_s1.wav")
    fn_list = glob(reg_fp)
    for idx, fn in enumerate(fn_list):
        logging.info(f"Processing {idx/len(fn_list)} ...")
        res = subprocess.run(["mv", fn, opt_dir],
                             capture_output=True, text=True,
                             encoding='utf-8')
        assert res.returncode == 0

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

