
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jieba_fast').setLevel(logging.WARNING)
logging.basicConfig(format='[%(asctime)s-%(levelname)s-%(funcName)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

import time
import os
import numpy as np
import soundfile as sf
import librosa
import torch
import LangSegment
from subprocess import getstatusoutput
from GSV_utils import cut5, process_text, merge_short_text_in_array, get_first, replace_consecutive_punctuation, clean_text_inf
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import SynthesizerTrn
from feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools.my_utils import load_audio
from module.mel_processing import spectrogram_torch
assert getstatusoutput("ls tools")[0] == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python GPT_SoVITS/GSV_model.py"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_GPT_MODEL_FP = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
DEFAULT_SOVITS_MODEL_FP = "GPT_SoVITS/pretrained_models/s2G488k.pth"
CNHUBERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
LANG_MAP = {"JP": "all_ja", "ZH": "all_zh", "EN": "en", "ZH_EN": "zh", "JP_EN": "ja", "AUTO": "auto"}
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
TMP_DIR = "./#tmp_output"
os.makedirs(TMP_DIR, exist_ok=True)


class ReferenceInfo:
    def __init__(self, audio_fp: str, text: str, lang: str):
        self.audio_fp = audio_fp
        self.text = text
        self.lang = lang


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class GSVModel:
    """
    @half: 半精度
    """
    def __init__(self, sovits_model_fp=None, gpt_model_fp=None, is_half=False, speaker="default"):
        self.sovits_model_fp = DEFAULT_SOVITS_MODEL_FP if sovits_model_fp is None else sovits_model_fp
        self.gpt_model_fp = DEFAULT_GPT_MODEL_FP if gpt_model_fp is None else gpt_model_fp
        self.is_half = is_half
        self.speaker = speaker
        logging.info(f">>> Will init GSVModel with:")
        logging.info(f"    [gpt_model_fp]: '{self.gpt_model_fp}'")
        logging.info(f"    [sovits_model_fp]: '{self.sovits_model_fp}'")
        logging.info(f">>> Initializing sovits ...")
        self.hps, self.vq_model = self.init_sovits()
        logging.info(f">>> Initializing gpt ...")
        self.hz, self.max_sec, self.t2s_model, self.config = self.init_gpt()
        logging.info(f">>> Initializing cnhubert ...")
        self.ssl_model = self.init_cnhubert()
        logging.info(f">>> Initializing bert ...")
        self.tokenizer, self.bert_model = self.init_bert()

        self.torch_dtype = torch.float16 if self.is_half else torch.float32


    def init_sovits(self):
        # >>> hps init
        dict_s2 = torch.load(self.sovits_model_fp, map_location="cpu")
        _hps = dict_s2["config"]
        _hps = DictToAttrRecursive(_hps)
        _hps.model.semantic_frame_rate = "25hz"
        _hps = _hps
        logging.debug(f">>> hps from sovits_model: {str(_hps)}\n")
        # >>> sovits init
        _vq_model = SynthesizerTrn(_hps.data.filter_length // 2 + 1,
                                   _hps.train.segment_size // _hps.data.hop_length,
                                   n_speakers=_hps.data.n_speakers,
                                   **_hps.model)
        if "pretrained" not in self.sovits_model_fp:
            del _vq_model.enc_q
        if self.is_half:
            _vq_model = _vq_model.half().to(DEVICE)
        else:
            _vq_model = _vq_model.to(DEVICE)
        _vq_model.eval()
        logging.debug(str(_vq_model.load_state_dict(dict_s2["weight"], strict=False)))
        return _hps, _vq_model

    def init_gpt(self):
        hz = 50
        dict_s1 = torch.load(self.gpt_model_fp, map_location="cpu")
        config = dict_s1["config"]
        logging.debug(f">>>  config from gpt: {str(config)}\n")
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(DEVICE)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        logging.debug("Number of parameter: %.2fM" % (total / 1e6))
        return hz, max_sec, t2s_model, config

    def init_cnhubert(self):
        cnhubert.cnhubert_base_path = CNHUBERT_MODEL_FP
        ssl_model = cnhubert.get_model()
        if self.is_half:
            ssl_model = ssl_model.half().to(DEVICE)
        else:
            ssl_model = ssl_model.to(DEVICE)
        return ssl_model

    def init_bert(self):
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_FP)
        bert_model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_FP)
        if self.is_half:
            bert_model = bert_model.half().to(DEVICE)
        else:
            bert_model = bert_model.to(DEVICE)
        return tokenizer, bert_model

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(DEVICE)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        # todo 看起来只有中文才提取了bert特征，试试加上英文bert
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(DEVICE)  # .to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.torch_dtype,
            ).to(DEVICE)

        return bert

    def get_phones_and_bert(self, text, language):
        if language in {"en", "all_zh", "all_ja"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = clean_text_inf(formattext, language)
            if language == "zh":
                bert = self.get_bert_feature(norm_text, word2ph).to(DEVICE)
            else:
                # todo 看起来只有中文才提取了bert特征，试试加上英文bert
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.torch_dtype,
                ).to(DEVICE)
        elif language in {"zh", "ja", "auto"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            logging.debug(f">>> textlist:\n {textlist}\n")
            logging.debug(f">>> langlist:\n {langlist}\n")
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        return phones, bert.to(self.torch_dtype), norm_text

    @staticmethod
    def get_spec(hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language,
                    top_k=20, top_p=0.6, temperature=0.6, ref_free=False, no_cut=False):
        
        # >>> format ref_text
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in SPLITS:
            prompt_text += "。" if prompt_language != "en" else "."
        logging.debug(f">>> Formatted Reference Text:\n'{prompt_text}'")
        # >>> format target_text
        text = text.strip("\n")
        text = replace_consecutive_punctuation(text)
        if text[0] not in SPLITS and len(get_first(text)) < 4:
            text = "。" + text if text_language != "en" else "." + text
        logging.debug(f">>> Formatted Target Text:\n'{text}'")
        
        zero_wav = np.zeros(int(self.hps.data.sampling_rate * 0.3),
                            dtype=np.float16 if self.is_half else np.float32)

        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half:
                wav16k = wav16k.half().to(DEVICE)
                zero_wav_torch = zero_wav_torch.half().to(DEVICE)
            else:
                wav16k = wav16k.to(DEVICE)
                zero_wav_torch = zero_wav_torch.to(DEVICE)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(DEVICE)

        # 不切，或直接按标点符号切
        text = text if no_cut else cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        logging.debug(f"实际输入的目标文本(切句后):\n{text}")
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language)
            

        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in SPLITS): text += "。" if text_language != "en" else "."
            logging.debug(f"实际输入的目标文本(每句):{text}")
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language)
            logging.debug(f"前端处理后的文本(每句):{norm_text2}")
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(DEVICE).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(DEVICE).unsqueeze(0)

            bert = bert.to(DEVICE).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(DEVICE)

            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )

            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self.get_spec(self.hps, ref_wav_path)  # .to(DEVICE)
            if self.is_half:
                refer = refer.half().to(DEVICE)
            else:
                refer = refer.to(DEVICE)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(DEVICE).unsqueeze(0), refer
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1: audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)

        yield self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

    # no_cut置为True时注意不要开头就打句号
    def predict(self, target_text, target_lang,
                ref_info: ReferenceInfo = None,
                top_k=1, top_p=0.8, temperature=0.8,
                ref_free: bool = False, no_cut: bool = False):
        # Synthesize audio
        synthesis_result = self.get_tts_wav(ref_wav_path=ref_info.audio_fp,
                                            prompt_text=ref_info.text,
                                            prompt_language=LANG_MAP[ref_info.lang],
                                            text=target_text,
                                            text_language=LANG_MAP[target_lang],
                                            top_k=top_k, top_p=top_p, temperature=temperature,
                                            ref_free=ref_free,
                                            no_cut=no_cut)

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            return last_sampling_rate, last_audio_data
        else:
            return None, None


if __name__ == '__main__':
    # sovits_model = "/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/XiaoLinShuo_e4_s60.pth"
    # gpt_model = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/XiaoLinShuo-e15.ckpt"
    # ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/Xu_Ran/vocals_as_reference.wav",
    #                           text="我想问一下，就是咱们那个疫情防控政策",
    #                           lang="ZH")

    sovits_model = "/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/XiaoLinShuo_e4_s60.pth"
    gpt_model = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/XiaoLinShuo-e15.ckpt"
    ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/XiaoLinShuo/denoised/2_vocals.wav_0005554240_0005717760.wav",
                              text="所以说啊这二十年真的是在斯大林的带领下，把前苏联的经济带上了一个新高度。",
                              lang="ZH")
    ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/XiaoLinShuo/denoised/2_vocals.wav_0004883520_0005045120.wav",
                              text="于是一九二八年之后，斯大林的前三个五年计划哈，那可谓是效果拔群。",
                              lang="ZH")

    M = GSVModel(sovits_model_fp=sovits_model, gpt_model_fp=gpt_model)

    sr, audio = M.predict(target_text="您好，我是您的个人助理！您可以问我今天天气如何。",
                          target_lang="ZH",
                          ref_info=ref_audio,
                          top_k=1, top_p=0, temperature=0,
                          ref_free=False, no_cut=True)
    sf.write(os.path.join(TMP_DIR, f"output_{time.time():.0f}.wav"), audio, sr)

    sr, audio = M.predict(target_text="我想问一下，关于极端天气防控的问题。",
                          target_lang="ZH",
                          ref_info=ref_audio,
                          top_k=1, top_p=0, temperature=0,
                          ref_free=False, no_cut=True)
    sf.write(os.path.join(TMP_DIR, f"output_{time.time():.0f}.wav"), audio, sr)

    sr, audio = M.predict(target_text="Hello sir, I'm your personal assistant. you can ask me about the whether.",
                          target_lang="EN",
                          ref_info=ref_audio,
                          top_k=1, top_p=0, temperature=0,
                          ref_free=False, no_cut=True)
    sf.write(os.path.join(TMP_DIR, f"output_ref_{time.time():.0f}_en.wav"), audio, sr)

    long_text = ("The sun rises, painting the sky with hues of gold and pink. The birds chirp merrily, "
                 "greeting the new day. A gentle breeze blows, carrying the fragrance of fresh flowers. "
                 "It's a beautiful start to another wonderful day.")
    for text in long_text.split("\\."):
        sr, audio = M.predict(target_text=text,
                              target_lang="EN",
                              ref_info=ref_audio,
                              top_k=1, top_p=0, temperature=0,
                              ref_free=False, no_cut=True)
        sf.write(os.path.join(TMP_DIR, f"output_ref_{time.time():.0f}_en.wav"), audio, sr)
