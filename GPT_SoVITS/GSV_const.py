import os
from GSV_model import ReferenceInfo

LANG_MAP = {"zh_cn": "ZH", "en_us": "EN"}
D_REF_SUFFIX = "default"
VOICE_SAMPLE_DIR = os.path.abspath(os.path.expanduser("./voice_sample/"))
GPT_DIR = os.path.abspath(os.path.expanduser("./GPT_weights/"))
SOVITS_DIR = os.path.abspath(os.path.expanduser("./SoVITS_weights/"))


class InferenceParam:
    trace_id: str = None
    speaker: str = None  # 角色音
    text: str = None  # 要合成的文本
    lang: str = None  # 合成音频的语言 (e.g. zh_cn/en_us)
    use_ref: bool = True  # 推理时是否使用参考音频的情绪，目前还不能置为False必须是True
    ref_suffix: str = D_REF_SUFFIX  # 当可提供多个参考音频时，指定参考音频的后缀
    nocut: bool = True  # 是否不做切分

    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        # {"JP": "all_ja", "ZH": "all_zh", "EN": "en", "ZH_EN": "zh", "JP_EN": "ja", "AUTO": "auto"}
        # lang = self.lang if self.lang in ["JP", "ZH", "EN", "ZH_EN", "JP_EN"] else "AUTO"
        lang = LANG_MAP.get(self.lang, "AUTO")
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
        return False if self.use_ref else True

    def __init__(self, info_dict):
        for key in self.__annotations__.keys():
            if key in info_dict:
                setattr(self, key, info_dict[key])
