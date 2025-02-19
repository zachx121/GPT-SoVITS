import logging
import os, sys

PROJ_DIR = os.path.abspath(os.path.join(__file__, "../../"))
print(f">>> PROJ_DIR: {PROJ_DIR}")

# if len(sys.argv) == 1: sys.argv.append('v2')
# version = "v1" if sys.argv[1] == "v1" else "v2"
version = "v2"
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings

warnings.filterwarnings("ignore")
import json, yaml, torch, pdb, re, shutil
import platform
import psutil
import wave
import librosa
import signal
from subprocess import Popen, getstatusoutput

torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if (os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if (name == "jieba.cache"): continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if (site_packages_roots == []): site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass
from tools import my_utils
import traceback
import shutil
import pdb
from subprocess import Popen
import signal
from config import python_exec, infer_device, is_half, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share

from multiprocessing import cpu_count
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
from . import GSV_const as C
from .GSV_const import Route as R
import utils_audio

logger = logging.getLogger()

n_cpu = cpu_count()

EXP_ROOT = "./logs"
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {"10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4",
                   "T4", "TITAN", "L4", "4060", "H"}
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    set_gpu_numbers.add(0)
    default_batch_size = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 2)
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if (int(input) not in set_gpu_numbers): return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","): output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


pretrained_sovits_name = ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                          "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

pretrained_model_list = (
    pretrained_sovits_name[-int(version[-1]) + 2], pretrained_sovits_name[-int(version[-1]) + 2].replace("s2G", "s2D"),
    pretrained_gpt_name[-int(version[-1]) + 2], "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base")

_ = ''
for i in pretrained_model_list:
    if os.path.exists(i):
        ...
    else:
        _ += f'\n    {i}'
if _:
    print("warning:", '以下模型不存在:' + _)

_ = [[], []]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    else:
        _[0].append("")  ##没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
    else:
        _[-1].append("")
pretrained_gpt_name, pretrained_sovits_name = _

SoVITS_weight_root = ["SoVITS_weights_v2", "SoVITS_weights"]
GPT_weight_root = ["GPT_weights_v2", "GPT_weights"]
for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name != ""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [name for name in pretrained_gpt_name if name != ""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()
for path in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(path, exist_ok=True)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {
        "choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid):
    if (system == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def log_audio_statistics(asr_dir):
    """ 打印音频数据的统计信息 """
    with open(os.path.join(asr_dir, "denoised.list"), "r") as fpr:
        lines = fpr.readlines()

    def _get_duration_of_wav(file_path):
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / rate
        return duration

    total_lines = len(lines)
    avg_words = sum([len(line.split("|")[3].split(" ")) for line in lines]) / total_lines
    max_words = max([len(line.split("|")[3].split(" ")) for line in lines])
    avg_duration = sum([_get_duration_of_wav(line.split("|")[0]) for line in lines]) / total_lines
    max_duration = max([_get_duration_of_wav(line.split("|")[0]) for line in lines])

    logger.info(f"""
        >>> 整体数据处理结果
        [total lines]: {total_lines}
        [avg words per line]: {avg_words}
        [max words per line]: {max_words}
        [avg audio duration]: {avg_duration}
        [max audio duration]: {max_duration}
    """)


def step_asr(denoised_dir, asr_dir, sid, lang="auto"):
    from tools.asr.config import asr_dict
    if lang.lower() in ["zh_cn"]:
        asr_py = asr_dict["达摩 ASR (中文)"]["path"]
        lang2funasrlang = {"zh_cn": "zh"}
        lang = lang2funasrlang[lang]
    else:
        asr_py = asr_dict["Faster Whisper (多语种)"]["path"]
        lang2whiserlang = {"en_us": "en", "jp_jp": "ja", "ko_kr": "ko", "auto":"auto"}
        lang = lang2whiserlang[lang]
    logger.info(f">>> Step_ASR's asr_py is '{asr_py}' using lang as '{lang}'")

    p_asr = None
    if (p_asr == None):
        # python tools/asr/fasterwhisper_asr.py -i ./voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz/denoised -o ./voice_sample/ChatTTS_Voice_Clone_0_Mike_yvmz/asr -s large -l en -p float32
        cmd = f"""python 'tools/asr/{asr_py}' \
        --input_folder {denoised_dir} \
        --output_folder {asr_dir} \
        --model_size large-v3-local \
        --language {lang.lower()} -p float32 \
        """
        logger.info(f"asr cmd: {cmd}")
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None

    fp = os.path.join(asr_dir, "denoised.list")
    min_duration, min_idx = 999, -1
    with open(os.path.join(fp), "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for idx, line in enumerate(lines):
            wav_file = line.split("|")[0]
            duration = librosa.get_duration(filename=os.path.join(PROJ_DIR, wav_file))
            if duration <= 3:
                continue
            elif duration <= min_duration:
                min_duration = duration
                min_idx = idx
            else:
                continue
    ref_line = lines.pop(min_idx)

    # 原文件最短的音频，放到另一个文件中作为参考音频使用
    cmd1 = f"""cp '{os.path.join(PROJ_DIR, ref_line.split("|")[0])}'  '{R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)}'"""
    cmd2 = f"""echo '{ref_line.split("|")[2]}|{ref_line.split("|")[3]}' > '{R.get_ref_text_fp(sid, C.D_REF_SUFFIX)}'"""
    logger.info(f">>> execute cmd1: {cmd1}")
    s, _ = getstatusoutput(cmd1)
    assert s == 0, f"cp audio fail. cmd: {cmd1}"
    logger.info(f">>> execute cmd1: {cmd2}")
    s, _ = getstatusoutput(cmd2)
    assert s == 0, f"echo file fail. cmd: {cmd2}"

    # 原文件最短的音频，从原文件中删掉
    with open(os.path.join(fp), "w", encoding="utf-8") as fw:
        fw.writelines(lines)

    if lang == "en_us":
        total_words = sum([len(l.split("|")[3].split(" ")) for l in lines])
        longest = max([len(l.split("|")[3].split(" ")) for l in lines])
        avg_words = total_words / len(lines)
    else:
        total_words = sum([len(l.split("|")[3]) for l in lines])
        longest = max([len(l.split("|")[3]) for l in lines])
        avg_words = total_words / len(lines)
    logger.info(f">>> ASR Finished. [Lines]: {len(lines)} [AvgWords]: {avg_words:.02f} [LongestSeg]: {longest}")
    log_audio_statistics(asr_dir)


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
    denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
    check_for_exists([denoise_inp_dir])
    cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
        python_exec, denoise_inp_dir, denoise_opt_dir, "float16" if is_half == True else "float32")

    # yield "语音降噪任务开启：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
    #                                                                               "visible": True}, {
    #     "__type__": "update"}, {"__type__": "update"}
    print(cmd)
    p_denoise = Popen(cmd, shell=True)
    p_denoise.wait()
    p_denoise = None
    # yield f"语音降噪任务完成, 查看终端进行下一步", {"__type__": "update", "visible": True}, {"__type__": "update",
    #                                                                                          "visible": False}, {
    #     "__type__": "update", "value": denoise_opt_dir}, {"__type__": "update", "value": denoise_opt_dir}


p_train_SoVITS = None


def open1Ba_vits(exp_name,
                 pretrained_s2G,
                 pretrained_s2D,
                 model_dir,
                 opt_dir,
                 batch_size=default_batch_size,
                 total_epoch=8, text_low_lr_rate=0.4, save_every_epoch=4,
                 if_save_latest=True, if_save_every_weights=True,
                 gpu_numbers1Ba="0"):
    global p_train_SoVITS
    with open("GPT_SoVITS/configs/s2.json") as f:
        data = f.read()
        data = json.loads(data)
    s2_dir = opt_dir
    # s2_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
    check_for_exists([s2_dir], is_train=True)
    if (is_half == False):
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers1Ba
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = model_dir
    data["name"] = exp_name
    data["version"] = version
    tmp_config_path = "%s/tmp_s2.json" % tmp
    with open(tmp_config_path, "w") as f:
        f.write(json.dumps(data))

    cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
    # yield "SoVITS训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
    #                                                                             "visible": True}
    print(cmd)
    p_train_SoVITS = Popen(cmd, shell=True)
    p_train_SoVITS.wait()
    p_train_SoVITS = None
    # yield "SoVITS训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


p_train_GPT = None


def open1Bb_gpt(exp_name,
                pretrained_s1,
                model_dir,
                opt_dir,
                batch_size=default_batch_size, total_epoch=15, save_every_epoch=5,
                if_dpo=False, if_save_latest=True, if_save_every_weights=True,
                gpu_numbers="0"):
    global p_train_GPT
    with open(
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    s1_dir = opt_dir
    # s1_dir = "%s/%s" % (exp_root, exp_name)
    os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
    check_for_exists([s1_dir], is_train=True)
    if (is_half == False):
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = model_dir
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1" % s1_dir
    # data["version"]=version

    os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(gpu_numbers.replace("-", ","))
    os.environ["hz"] = "25hz"
    tmp_config_path = "%s/tmp_s1.yaml" % tmp
    with open(tmp_config_path, "w") as f:
        f.write(yaml.dump(data, default_flow_style=False))
    # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
    cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
    # yield "GPT训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    print(cmd)
    p_train_GPT = Popen(cmd, shell=True)
    p_train_GPT.wait()
    p_train_GPT = None
    # yield "GPT训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


def step_convert2wav(inp):
    for name in sorted(list(os.listdir(inp))):
        if any(name.endswith(i) for i in ['m4a', 'mp3', 'mp4']):
            # inp = "/root/GPT-SoVITS/voice_sample/ChatTTS_Voice_Clone_4_222rb2j"
            fp = os.path.join(inp, name)
            new_fp = os.path.join(inp, os.path.splitext(name)[0] + ".wav")
            cmd = f"ffmpeg -y -i {fp} {new_fp}"
            logger.info(f">>> convert2wav: '{cmd}'")
            s, _ = getstatusoutput(cmd)
            assert s == 0, f"ffmpeg转换格式时错误, cmd: '{cmd}'"
            s, _ = getstatusoutput(f"rm {fp}")
            assert s == 0, f"删除文件失败, cmd: 'rm {fp}'"


ps_slice = []


def open_slice(inp, opt_root,
               min_interval=100,  # 最短切割间隔 (说话快的人就可以调小一点，默认是300）
               threshold=-34,  # 音量小于这个值视作静音的备选切割点
               min_length=4000,  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
               hop_size=10,  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
               max_sil_kept=500,  # 切完后静音最多留多长
               _max=0.9,  # 归一化后最大值多少
               alpha=0.25,  # 混多少比例归一化后音频进来
               n_parts=1  # 并行数
               ):
    logger.info(f"inp: {inp} opt_root: {opt_root}")
    global ps_slice
    for i_part in range(n_parts):
        cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (
            python_exec, inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha,
            i_part, n_parts)
        print(cmd)
        p = Popen(cmd, shell=True)
        ps_slice.append(p)

    for p in ps_slice:
        p.wait()
    ps_slice = []


ps1a=[]
def open1a(inp_text,
           inp_wav_dir,
           exp_name,
           bert_pretrained_dir,
           gpu_numbers="0-0",
           ):
    global ps1a
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text,inp_wav_dir], is_dataset_processing=True)
    opt_dir="%s/%s"%(EXP_ROOT, exp_name)
    config={
        "inp_text":inp_text,
        "inp_wav_dir":inp_wav_dir,
        "exp_name":exp_name,
        "opt_dir":opt_dir,
        "bert_pretrained_dir":bert_pretrained_dir,
    }
    gpu_names=gpu_numbers.split("-")
    all_parts=len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                "is_half": str(is_half)
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1a.append(p)
    logger.info("文本进程执行中")
    # yield "文本进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    for p in ps1a:
        p.wait()
    opt = []
    for i_part in range(all_parts):
        txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    path_text = "%s/2-name2text.txt" % opt_dir
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    ps1a=[]
    if len("".join(opt)) > 0:
        logger.info("文本进程成功")
        # yield "文本进程成功", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        logger.info("文本进程失败")
        # yield "文本进程失败", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


ps1b=[]
def open1b(inp_text,inp_wav_dir,exp_name,
           ssl_pretrained_dir,
           gpu_numbers="0-0"):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text,inp_wav_dir], is_dataset_processing=True)
    config={
        "inp_text":inp_text,
        "inp_wav_dir":inp_wav_dir,
        "exp_name":exp_name,
        "opt_dir":"%s/%s"%(EXP_ROOT, exp_name),
        "cnhubert_base_dir":ssl_pretrained_dir,
        "is_half": str(is_half)
}
    gpu_names=gpu_numbers.split("-")
    all_parts=len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1b.append(p)
    # yield "SSL提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    logger.info("SSL提取进程执行中")
    for p in ps1b:
        p.wait()
    ps1b=[]
    # yield "SSL提取进程结束", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}
    logger.info("SSL提取进程结束")


ps1c=[]
def open1c(inp_text,exp_name,pretrained_s2G_path,
           gpu_numbers="0-0"):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    check_for_exists([inp_text,''], is_dataset_processing=True)
    opt_dir="%s/%s"%(EXP_ROOT, exp_name)
    config={
        "inp_text":inp_text,
        "exp_name":exp_name,
        "opt_dir":opt_dir,
        "pretrained_s2G":pretrained_s2G_path,
        "s2config_path":"GPT_SoVITS/configs/s2.json",
        "is_half": str(is_half)
    }
    gpu_names=gpu_numbers.split("-")
    all_parts=len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1c.append(p)
    logger.info("语义token提取进程执行中")
    # yield "语义token提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    for p in ps1c:
        p.wait()
    opt = ["item_name\tsemantic_audio"]
    path_semantic = "%s/6-name2semantic.tsv" % opt_dir
    for i_part in range(all_parts):
        semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
        with open(semantic_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(semantic_path)
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    ps1c=[]
    logger.info("语义token提取进程结束")
    # yield "语义token提取进程结束", {"__type__":"update","visible":True}, {"__type__":"update","visible":False}



#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc = []
def open1abc(inp_text, inp_wav_dir, exp_name, opt_dir,
             bert_pretrained_dir,
             ssl_pretrained_dir,
             pretrained_s2G_path,
             gpu_numbers1a="0-0",
             gpu_numbers1Ba="0-0",
             gpu_numbers1c="0-0",
             ):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text, inp_wav_dir])
    # opt_dir = "%s/%s" % (exp_root, exp_name)
    #############################1a
    path_text = "%s/2-name2text.txt" % opt_dir
    config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "bert_pretrained_dir": bert_pretrained_dir,
        "is_half": str(is_half)
    }
    gpu_names = gpu_numbers1a.split("-")
    all_parts = len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1abc.append(p)
    # yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    logger.info("进度：1a-ing")
    for p in ps1abc: p.wait()

    opt = []
    for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
        txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
    # yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    logger.info("进度：1a-done")
    ps1abc = []
    #############################1b
    config = {
        "inp_text": inp_text,
        "inp_wav_dir": inp_wav_dir,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "cnhubert_base_dir": ssl_pretrained_dir,
    }
    gpu_names = gpu_numbers1Ba.split("-")
    all_parts = len(gpu_names)
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1abc.append(p)
    logger.info("进度：1a-done, 1b-ing")
    # yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update",
    #                                                                          "visible": True}
    for p in ps1abc: p.wait()
    logger.info("进度：1a1b-done")
    # yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    ps1abc = []
    #############################1c
    path_semantic = "%s/6-name2semantic.tsv" % opt_dir
    config = {
        "inp_text": inp_text,
        "exp_name": exp_name,
        "opt_dir": opt_dir,
        "pretrained_s2G": pretrained_s2G_path,
        "s2config_path": "GPT_SoVITS/configs/s2.json",
    }
    gpu_names = gpu_numbers1c.split("-")
    all_parts = len(gpu_names)
    logger.warning(f"all_parts: {all_parts}")
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
            }
        )
        os.environ.update(config)
        logger.warning(os.environ.get("pretrained_s2G"))
        logger.warning(os.environ.get("s2config_path"))
        logger.warning(os.environ.get("version", "v2"))
        cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
        print(cmd)
        p = Popen(cmd, shell=True)
        ps1abc.append(p)
    logger.info("进度：1a1b-done, 1cing")
    # yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update",
    #                                                                           "visible": True}
    for p in ps1abc: p.wait()

    opt = ["item_name\tsemantic_audio"]
    for i_part in range(all_parts):
        semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
        with open(semantic_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(semantic_path)
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    logger.info("进度：all-done")
        # yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
    ps1abc = []
    # yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


def check_for_exists(file_list=None, is_train=False, is_dataset_processing=False):
    missing_files = []
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0], '2-name2text.txt'))
        file_list.append(os.path.join(file_list[0], '3-bert'))
        file_list.append(os.path.join(file_list[0], '4-cnhubert'))
        file_list.append(os.path.join(file_list[0], '5-wav32k'))
        file_list.append(os.path.join(file_list[0], '6-name2semantic.tsv'))
    for file in file_list:
        if os.path.exists(file):
            pass
        else:
            missing_files.append(file)
    if missing_files:
        if is_train:
            for missing_file in missing_files:
                if missing_file != '':
                    logger.warning(f"miss file: {missing_file}")
        else:
            for missing_file in missing_files:
                if missing_file != '':
                    logger.warning(f"miss file: {missing_file}")
            if file_list[-1] == [''] and is_dataset_processing:
                pass
            else:
                logger.warning(f"miss file: {','.join(file_list)}")


def workflow(inp_params):
    sid = inp_params["speaker_id"]
    lang = inp_params["lang"]
    data_urls = inp_params["data_urls"]
    post2oss = inp_params.get("post2oss", "1")

    assert lang in ["en_us", "jp_jp", "ko_kr", "zh_cn", "auto"], "语言必须是 en_us/jp_jp/ko_kr/zh_cn/auto"

    inp_dir = os.path.join(C.VOICE_SAMPLE_DIR, sid)  # 使用 C 中的 VOICE_SAMPLE_DIR
    logger.info(f">>> Start with ExpName='{sid}', InputDir='{inp_dir}', Language='{lang}'")
    logger.info(f">>> data_urls: {data_urls}")

    SLICE_DIR = os.path.join(inp_dir, 'sliced')
    DENOISED_DIR = os.path.join(inp_dir, 'denoised')
    ASR_DIR = os.path.join(inp_dir, 'asr')
    ASR_RES_FP = os.path.join(ASR_DIR, os.path.basename(DENOISED_DIR)) + ".list"
    PRETRAIN_OPT_DIR = os.path.join(inp_dir, "logs")
    SoVITS_weight_root = os.path.join(C.SOVITS_DIR, sid)
    GPT_weight_root = os.path.join(C.GPT_DIR, sid)

    if True:  # skip
        if os.path.exists(inp_dir):
            shutil.rmtree(inp_dir)
        os.makedirs(inp_dir, exist_ok=True)

        for i in [SLICE_DIR, DENOISED_DIR, ASR_DIR, SoVITS_weight_root, GPT_weight_root]:
            os.makedirs(i, exist_ok=True)

        # Data Process
        logger.info(f">>> Start Data Preparing. saved at {inp_dir}")
        utils_audio.download_files_in_parallel(data_urls, inp_dir)
        logger.info(">>> At step_convert2wav")
        step_convert2wav(inp_dir)
        logger.info(f">>> At step_slice. saved at {SLICE_DIR}")
        open_slice(inp_dir, SLICE_DIR, min_interval=80 if lang.lower() in ["en_us"] else 300)
        logger.info(f">>> At step_denoise. saved at {DENOISED_DIR}")
        open_denoise(SLICE_DIR, DENOISED_DIR)
        logger.info(f">>> At step_asr. saved at {ASR_DIR}")
        step_asr(DENOISED_DIR, ASR_DIR, sid, lang)

        # Duplicated
        # logger.info(">>> At Pretrain Application | open1a")
        # open1a(ASR_RES_FP, DENOISED_DIR, sid, bert_pretrained_dir=C.PRETRAIN_BERT_DIR)
        # logger.info(">>> At Pretrain Application | open1b")
        # open1b(ASR_RES_FP, DENOISED_DIR, sid, ssl_pretrained_dir=C.PRETRAIN_SSL_DIR)
        # logger.info(">>> At Pretrain Application | open1c")
        # open1c(ASR_RES_FP, sid, pretrained_s2G_path=C.PRETRAIN_S2G_FP)

        # Pretrain Application
        open1abc(ASR_RES_FP, DENOISED_DIR, sid,
                 opt_dir=PRETRAIN_OPT_DIR,
                 bert_pretrained_dir=C.PRETRAIN_BERT_DIR,
                 ssl_pretrained_dir=C.PRETRAIN_SSL_DIR,
                 pretrained_s2G_path=C.PRETRAIN_S2G_FP)

        # Model Training
        logger.info(">>> At Model Training")
        open1Ba_vits(sid, pretrained_s2G=C.PRETRAIN_S2G_FP, pretrained_s2D=C.PRETRAIN_S2D_FP,
                     model_dir=SoVITS_weight_root, opt_dir=PRETRAIN_OPT_DIR)
        open1Bb_gpt(sid, pretrained_s1=C.PRETRAIN_GPT_FP, model_dir=GPT_weight_root, opt_dir=PRETRAIN_OPT_DIR)

        # Update latest model name
        sovits_fp = os.path.join(SoVITS_weight_root, sid + ".latest.pth")
        gpt_fp = os.path.join(GPT_weight_root, sid + ".latest.ckpt")
        os.rename(os.path.join(SoVITS_weight_root, utils_audio.get_latest_fp(SoVITS_weight_root)), sovits_fp)
        os.rename(os.path.join(GPT_weight_root, utils_audio.get_latest_fp(GPT_weight_root)), gpt_fp)

    if post2oss=="1":
        logger.info(">>> Uploading sovits model to qiniu.")
        url = utils_audio.post2qiniu(sovits_fp, R.get_sovits_osskey(sid))
        logger.info(f">>> url as: '{url}'")
        logger.info(">>> Uploading gpt model to qiniu.")
        url = utils_audio.post2qiniu(gpt_fp, R.get_gpt_osskey(sid))
        logger.info(f">>> url as: '{url}'")

        logger.info(">>> Uploading default_ref to qiniu.")
        audio_url = utils_audio.post2qiniu(R.get_ref_audio_fp(sid, C.D_REF_SUFFIX), R.get_ref_audio_osskey(sid))
        text_url = utils_audio.post2qiniu(R.get_ref_text_fp(sid, C.D_REF_SUFFIX), R.get_ref_text_osskey(sid))
        logging.info(f">>> [audio_url]:'{audio_url}' [text_url]:'{text_url}'")


# python -m service_GSV.GSV_train_standalone test_cxm zh_cn 'model/clone/device/20250211/1000294265/6f50a0eb-2a46-4973-93d2-2dbe84d0f3a7.m4a'
if __name__ == '__main__':
    try:
        assert len(sys.argv) >= 4, "python -m service_GSV.GSV_train_standalone <sid> <lang> <data_urls(逗号拼接)>"
        sid = sys.argv[1]
        lang = sys.argv[2]
        data_urls = sys.argv[3].split(",")
        params = {"speaker_id": sid,
                  "lang": lang,
                  "data_urls": data_urls}

        # url1 = utils_audio.get_url_from_qiniu("model/clone/device/20250211/1000294265/6f50a0eb-2a46-4973-93d2-2dbe84d0f3a7.m4a")
        # params = {"speaker_id": "test_cxm",
        #           "lang": "zh_cn",
        #           "data_urls": [url1]}
        workflow(params)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in task: {e}", exc_info=True)
        sys.exit(1)



