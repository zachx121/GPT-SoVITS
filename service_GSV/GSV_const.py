import os

import pylab as p


class ReferenceInfo:
    def __init__(self, audio_fp: str, text: str, lang: str):
        self.audio_fp = audio_fp
        self.text = text
        self.lang = lang

    @staticmethod
    def from_sid(sid):
        import subprocess
        ref_audio_fp = Route.get_ref_audio_fp(sid, D_REF_SUFFIX)
        ref_lang, ref_text = subprocess.getoutput(f"cat {Route.get_ref_text_fp(sid, D_REF_SUFFIX)}").strip().split("|")
        ref_info = ReferenceInfo(audio_fp=ref_audio_fp,
                                 text=ref_text,
                                 lang=ref_lang)
        return ref_info

    def __str__(self):
        return f"audio_fp='{self.audio_fp}', text='{self.text}', lang='{self.lang}'"


# 外部输入的语言参数转换为GSV框架内默认的语言参数
LANG_MAP = {"EN": "en", "en_us": "en",
            "JP": "all_ja", "jp_jp": "all_ja",
            "KO": "all_ko", "ko_kr": "all_ko",
            "ZH": "all_zh", "zh_cn": "all_zh",
            "YUE": "all_yue",
            "ZH_EN": "zh",  # 中英混合识别
            "JP_EN": "ja",  # 日英混合识别
            "YUE_EN": "yue",  # 粤英混合识别
            "KO_EN": "ko",  # 韩英混合识别
            "AUTO": "auto",
            "AUTO_YUE": "auto_yue"
            }
D_REF_SUFFIX = "default"
# LOG_DIR = "logs"
# VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("./voice_sample/"))
# GPT_DIR = os.path.abspath(os.path.expanduser("./GPT_weights/"))
# SOVITS_DIR = os.path.abspath(os.path.expanduser("./SoVITS_weights/"))
LOG_DIR = "/root/autodl-tmp/logs"
VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("/root/autodl-fs/voice_sample/"))
GPT_DIR = os.path.abspath(os.path.expanduser("/root/autodl-fs/GPT_weights/"))
SOVITS_DIR = os.path.abspath(os.path.expanduser("/root/autodl-fs/SoVITS_weights/"))

PRETRAIN_BERT_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
PRETRAIN_SSL_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
PRETRAIN_S2G_FP = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
PRETRAIN_S2D_FP = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
PRETRAIN_GPT_FP = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"


class Route:
    @staticmethod
    def get_sovits_fp(sid):
        return os.path.join(SOVITS_DIR, sid, sid + ".latest.pth")

    @staticmethod
    def get_gpt_fp(sid):
        return os.path.join(GPT_DIR, sid, sid + ".latest.ckpt")

    @staticmethod
    def get_sovits_osskey(sid):
        return f"models/{sid}/{sid}_sovits"

    @staticmethod
    def get_gpt_osskey(sid):
        return f"models/{sid}/{sid}_gpt"

    @staticmethod
    def get_ref_audio_osskey(sid):
        return f"models/{sid}/{sid}_ref_audio_default.wav"

    @staticmethod
    def get_ref_text_osskey(sid):
        return f"models/{sid}/{sid}_ref_text_default.txt"

    @staticmethod
    def get_ref_audio_fp(sid, suffix):
        return os.path.join(VOICE_SAMPLE_DIR, sid, f'ref_audio_{suffix}.wav')

    @staticmethod
    def get_ref_text_fp(sid, suffix):
        return os.path.join(VOICE_SAMPLE_DIR, sid, f'ref_text_{suffix}.txt')


class InferenceParam:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. zh_cn/en_us)
    use_ref: bool = None  # 推理时是否使用参考音频的情绪，目前还不能置为False必须是True
    ref_suffix: str = D_REF_SUFFIX  # 当可提供多个参考音频时，指定参考音频的后缀
    nocut: bool = True  # 是否不做切分
    debug: bool = False
    result_queue_name: str = None # 请求返回结果的队列名
    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        # {"JP": "all_ja", "ZH": "all_zh", "EN": "en", "ZH_EN": "zh", "JP_EN": "ja", "AUTO": "auto"}
        # lang = self.lang if self.lang in ["JP", "ZH", "EN", "ZH_EN", "JP_EN"] else "AUTO"
        lang = LANG_MAP.get(self.lang, "auto")
        return lang

    @property
    def ref_info(self):
        if self.ref_free:
            return ReferenceInfo(audio_fp="", text="", lang="")
        ref_dir = os.path.join(VOICE_SAMPLE_DIR, self.speaker)
        # ~/tmp/ref_audio.wav
        audio_fp = os.path.join(ref_dir, f'ref_audio_{self.ref_suffix}.wav')
        # ~/tmp/ref_text.txt 里面必须是竖线分割的<语言>|<文本> e.g."ZH|语音的具体文本内容。"
        text_fp = os.path.join(ref_dir, f'ref_text_{self.ref_suffix}.txt')
        assert os.path.exists(audio_fp)
        assert os.path.exists(text_fp)
        with open(text_fp, 'r', encoding='utf - 8') as fr:
            lang, text = fr.readlines()[0].split("|")
        # lang, text = getstatusoutput(f"cat '{text_fp}'")[1].encode().split("|")
        return ReferenceInfo(audio_fp=audio_fp, text=text, lang=lang)

    @property
    def ref_free(self):
        return not self.use_ref if self.use_ref is not None else None

    def __init__(self, info_dict):
        for key in self.__annotations__.keys():
            if key in info_dict and info_dict[key] is not None:
                setattr(self, key, info_dict[key])
                
    def __repr__(self):
        return (f"InferenceParam(trace_id={self.trace_id}, speaker={self.speaker}, "
                f"text={self.text}, lang={self.lang}, use_ref={self.use_ref}, "
                f"ref_suffix={self.ref_suffix}, nocut={self.nocut}, debug={self.debug}),result_queue_name={self.result_queue_name}")



