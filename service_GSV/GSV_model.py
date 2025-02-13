
import logging
import sys


logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jieba_fast').setLevel(logging.WARNING)
logging.basicConfig(format='[%(asctime)s-%(levelname)s-%(funcName)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

import re
import time
from time import time as ttime
import os
import numpy as np
import soundfile as sf
import json
import librosa
import torch
import traceback
import LangSegment
from subprocess import getstatusoutput
from GSV_utils import cut5, process_text, merge_short_text_in_array, get_first, replace_consecutive_punctuation, clean_text_inf
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools.my_utils import load_audio
from GPT_SoVITS.module.mel_processing import spectrogram_torch
assert getstatusoutput("ls tools")[0] == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python GPT_SoVITS/GSV_model.py"

from GPT_SoVITS.text import chinese
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_GPT_MODEL_FP = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
DEFAULT_SOVITS_MODEL_FP = "GPT_SoVITS/pretrained_models/s2G488k.pth"
CNHUBERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
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
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
TMP_DIR = "./#tmp_output"
os.makedirs(TMP_DIR, exist_ok=True)


class ReferenceInfo:
    def __init__(self, audio_fp: str, text: str, lang: str):
        self.audio_fp = audio_fp
        self.text = text
        self.lang = lang

    def __str__(self):
        return f"audio_fp='{self.audio_fp}', text='{self.text}', lang='{self.lang}'"


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
        self.ssl_model = self.init_cnhubert_sslmodel()
        logging.info(f">>> Initializing bert ...")
        self.tokenizer, self.bert_model = self.init_bert()

        self.torch_dtype = torch.float16 if self.is_half else torch.float32


    # api.py/change_sovits_weights
    def init_sovits(self):
        # >>> hps init
        global vq_model, hps
        global vq_model, hps, version
        dict_s2 = torch.load(self.sovits_model_fp, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        version = hps.model.version
        # print("sovits版本:",hps.model.version)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if ("pretrained" not in self.sovits_model_fp):
            del vq_model.enc_q
        if self.is_half == True:
            vq_model = vq_model.half().to(DEVICE)
        else:
            vq_model = vq_model.to(DEVICE)
        vq_model.eval()
        print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

        with open("./weight.json") as f:
            data = f.read()
            data = json.loads(data)
            data["SoVITS"][version] = self.sovits_model_fp
        with open("./weight.json", "w") as f:
            f.write(json.dumps(data))
        # if prompt_language is not None and text_language is not None:
        #     if prompt_language in list(dict_language.keys()):
        #         prompt_text_update, prompt_language_update = {'__type__': 'update'}, {'__type__': 'update',
        #                                                                               'value': prompt_language}
        #     else:
        #         prompt_text_update = {'__type__': 'update', 'value': ''}
        #         prompt_language_update = {'__type__': 'update', 'value': i18n("中文")}
        #     if text_language in list(dict_language.keys()):
        #         text_update, text_language_update = {'__type__': 'update'}, {'__type__': 'update',
        #                                                                      'value': text_language}
        #     else:
        #         text_update = {'__type__': 'update', 'value': ''}
        #         text_language_update = {'__type__': 'update', 'value': i18n("中文")}
        #     # return {'__type__': 'update', 'choices': list(dict_language.keys())}, {'__type__': 'update',
        #     #                                                                        'choices': list(
        #     #                                                                            dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update
        return hps, vq_model

    def init_gpt(self):
        global hz, max_sec, t2s_model, config
        hz = 50
        dict_s1 = torch.load(self.gpt_model_fp, map_location="cpu")
        config = dict_s1["config"]
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(DEVICE)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        with open("./weight.json") as f:
            data = f.read()
            data = json.loads(data)
            data["GPT"][version] = self.gpt_model_fp
        with open("./weight.json", "w") as f: f.write(json.dumps(data))

        return hz, max_sec, t2s_model, config

    def init_cnhubert_sslmodel(self):
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
                dtype=torch.float16 if self.is_half == True else torch.float32,
            ).to(DEVICE)

        return bert

    def get_phones_and_bert(self, text,language,version="v2"):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(DEVICE)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half == True else torch.float32,
                ).to(DEVICE)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        return phones, bert.to(self.torch_dtype), norm_text

    @staticmethod
    def clean_text_inf(text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    @staticmethod
    def get_spepc(hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if (maxx > 1): audio /= min(2, maxx)
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



    def get_tts_wav(self, text, text_language,
                    ref_wav_path, prompt_text, prompt_language,
                    no_cut=False,
                    top_k=20, top_p=1.0, temperature=0.99, speed=1,
                    ref_free=False, if_freeze=False,
                    inp_refs=None, cache=None):
        cache = {} if cache is None else cache
        inp_refs = [] if inp_refs is None else inp_refs  # 多个参考音频文件（建议同性），平均融合他们的音色
        t = []
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        t0 = ttime()
        # 外部传参时已经转换好了
        # prompt_language = dict_language[prompt_language]
        # text_language = dict_language[text_language]

        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in SPLITS): prompt_text += "。" if prompt_language != "en" else "."
            logging.info(f"实际输入的参考文本: {prompt_text}")
        text = text.strip("\n")
        if (text[0] not in SPLITS and len(
            get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        logging.info(f"实际输入的目标文本: {text}")
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )
        if not ref_free:
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
                codes = vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]
                prompt = prompt_semantic.unsqueeze(0).to(DEVICE)

        t1 = ttime()
        t.append(t1 - t0)

        text = text if no_cut else cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        logging.info(f"实际输入的目标文本(切句后): {text}")
        texts = text.split("\n")
        texts = process_text(texts)
        texts = merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language, version)

        for i_text, text in enumerate(texts):
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in SPLITS): text += "。" if text_language != "en" else "."
            logging.info(f"实际输入的目标文本(每句): {text}")
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, version)
            logging.info(f"前端处理后的文本(每句): {norm_text2}")
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(DEVICE).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(DEVICE).unsqueeze(0)

            bert = bert.to(DEVICE).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(DEVICE)

            t2 = ttime()
            # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
            # print(cache.keys(),if_freeze)
            if (i_text in cache and if_freeze == True):
                pred_semantic = cache[i_text]
            else:
                with torch.no_grad():
                    pred_semantic, idx = t2s_model.model.infer_panel(
                        all_phoneme_ids,
                        all_phoneme_len,
                        None if ref_free else prompt,
                        bert,
                        # prompt_phone_len=ph_offset,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        early_stop_num=hz * max_sec,
                    )
                    pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                    cache[i_text] = pred_semantic
            t3 = ttime()
            refers = []
            for path in inp_refs:
                try:
                    refer = self.get_spepc(hps, path.name).to(self.torch_dtype).to(DEVICE)
                    refers.append(refer)
                except:
                    traceback.print_exc()
            if (len(refers) == 0): refers = [self.get_spepc(hps, ref_wav_path).to(self.torch_dtype).to(DEVICE)]
            audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(DEVICE).unsqueeze(0), refers,
                                     speed=speed).detach().cpu().numpy()[0, 0])
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1: audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
            t.extend([t2 - t1, t3 - t2, t4 - t3])
            t1 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" %
              (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
              )
        yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

    # no_cut置为True时注意不要开头就打句号
    def predict(self, target_text, target_lang,
                ref_info: ReferenceInfo = None,
                top_k=20, top_p=1.0, temperature=0.99, speed=1,
                ref_free: bool = False, no_cut: bool = False):
        # Synthesize audio
        synthesis_result = self.get_tts_wav(text=target_text, text_language=LANG_MAP[target_lang],
                                            ref_wav_path=ref_info.audio_fp,
                                            prompt_text=ref_info.text, prompt_language=LANG_MAP[ref_info.lang],
                                            top_k=top_k, top_p=top_p, temperature=temperature, speed=speed,
                                            ref_free=ref_free, no_cut=no_cut)

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            return last_sampling_rate, last_audio_data
        else:
            return None, None


if __name__ == '__main__':
    sovits_model = "/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/x_e8_s96.pth"
    gpt_model = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/x-e15.ckpt"
    ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/xn/vocals_as_reference.wav",
                              text="我想问一下，就是咱们那个疫情防控政策",
                              lang="ZH")
    M = GSVModel(sovits_model_fp=sovits_model, gpt_model_fp=gpt_model)

    sr, audio = M.predict(target_text="c测试",
                          target_lang="ZH",
                          ref_info=ref_audio,
                          top_k=1, top_p=0, temperature=0,
                          ref_free=False, no_cut=True)
    sf.write(os.path.join("/Users/bytedance/Downloads", f"output_{time.time():.0f}.wav"), audio, sr)

    sys.exit(0)

    # sovits_model = "/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/XiaoLinShuo_e4_s60.pth"
    # gpt_model = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/XiaoLinShuo-e15.ckpt"
    # ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/XiaoLinShuo/denoised/2_vocals.wav_0005554240_0005717760.wav",
    #                           text="所以说啊这二十年真的是在斯大林的带领下，把前苏联的经济带上了一个新高度。",
    #                           lang="ZH")
    # ref_audio = ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/XiaoLinShuo/denoised/2_vocals.wav_0004883520_0005045120.wav",
    #                           text="于是一九二八年之后，斯大林的前三个五年计划哈，那可谓是效果拔群。",
    #                           lang="ZH")

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
