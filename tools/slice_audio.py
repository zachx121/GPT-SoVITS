import os,sys,numpy as np
import traceback
from scipy.io import wavfile
# parent_directory = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(parent_directory)
from tools.my_utils import load_audio
from slicer2 import Slicer
from subprocess import getstatusoutput

def slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,i_part,all_part):
    os.makedirs(opt_root,exist_ok=True)
    if os.path.isfile(inp):
        input=[inp]
    elif os.path.isdir(inp):
        input=[os.path.join(inp, name) for name in sorted(list(os.listdir(inp))) if name.endswith(".wav")]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    assert len(input) > 0
    print(">>> All input as follow:\n%s" % '\n'.join(input))
    sr = 32000  # 长音频采样率
    slicer = Slicer(
        sr=sr,  # 长音频采样率
        threshold=      int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=     int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=   int(min_interval),  # 最短切割间隔
        hop_size=       int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=   int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max=float(_max)
    alpha=float(alpha)
    for inp_path in input[int(i_part)::int(all_part)]:
        # print(inp_path)
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, sr)
            # print(audio.shape)
            res = slicer.slice(audio)
            print(f"总共切分成: {len(res)} 最长时间: {max([i[0].shape[0] / sr for i in res])}")
            for chunk, start, end in res:  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if(tmp_max>1):chunk/=tmp_max
                chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                    sr,
                    # chunk.astype(np.float32),
                    (chunk * 32767).astype(np.int16),
                )
        except:
            print(inp_path,"->fail->",traceback.format_exc())
    return "执行完毕，请检查输出文件"

print(slice(*sys.argv[1:]))

