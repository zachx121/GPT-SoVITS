
import logging
import sys

#a
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jieba_fast').setLevel(logging.WARNING)
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

import re
import time
from time import time as ttime
import os
import shutil
import numpy as np
import soundfile as sf
import json
import librosa
import torch
import traceback
import LangSegment
import gradio as gr
from subprocess import getstatusoutput
from . import GSV_const as C
from .GSV_const import Route as R
from .GSV_const import ReferenceInfo
from .GSV_utils import cut5, process_text, merge_short_text_in_array, get_first, replace_consecutive_punctuation, clean_text_inf
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools.my_utils import load_audio
from GPT_SoVITS.module.mel_processing import spectrogram_torch
assert getstatusoutput("ls tools")[0] == 0, "必须在项目根目录下执行不然会有路径问题 e.g. python -m GPT_SoVITS.GSV_model"

from GPT_SoVITS.text import chinese
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_GPT_MODEL_FP = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
DEFAULT_SOVITS_MODEL_FP = "GPT_SoVITS/pretrained_models/s2G488k.pth"
CNHUBERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_MODEL_FP = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

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
        self.ref_info, self.ref_wav_int16, self.ref_sr = None, None, None


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
        else:
            raise Exception(f"GSV_model内部使用了异常语言: '{language}'")
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
            logging.debug(f"实际输入的参考文本: {prompt_text}")
        text = text.strip("\n")
        if (text[0] not in SPLITS and len(
            get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

        logging.debug(f"实际输入的目标文本: {text}")
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
        logging.debug(f"实际输入的目标文本(切句后): {text}")
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
            logging.debug(f"实际输入的目标文本(每句): {text}")
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language, version)
            logging.debug(f"前端处理后的文本(每句): {norm_text2}")
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

    @staticmethod
    def is_pure_noise(wav_arr, threshold_std=0.01, threshold_entropy=0.9, debug=False, auto_cast=True, **kwargs):
        """
            noise_list=[]
            for fn in os.listdir("/Users/zhou/Downloads/abcaudio"):
                if any(fn.endswith(suffix) for suffix in ['wav','pcm','m4a','mp3']):
                    print(f">>> processing {fn}")
                    wav_arr, wav_sr = librosa.load(os.path.join("/Users/zhou/Downloads/abcaudio", fn))
                    # wav_arr = (np.clip(wav_arr, -1.0, 1.0) * 32767).astype(np.int16)
                    res = is_pure_noise(wav_arr, debug=True)
                    if res == True:
                        print(f">>>>>> 检测到纯噪音 {fn}<<<<<<")
                        noise_list.append(fn)
                else:
                    continue
            print(noise_list)
        """

        if auto_cast and wav_arr.dtype == np.int16:
            wav_arr = wav_arr.astype(np.float32) / 32768.0  # 归一化到 [-1.0, 1.0]

        assert wav_arr.dtype == np.float32
        # 1. 计算方差（标准化后）
        y_norm = wav_arr / np.max(np.abs(wav_arr))
        std_dev = np.std(y_norm)

        # 2. 计算谱熵
        spec = np.abs(librosa.stft(wav_arr))
        spec_norm = spec / np.sum(spec)
        entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-10))
        normalized_entropy = entropy / np.log2(spec.shape[0] * spec.shape[1])

        # 3. 计算信噪比估计值 (使用能量比作为估计)
        signal_power = np.mean(wav_arr ** 2)
        noise_power_estimate = np.var(wav_arr - librosa.effects.preemphasis(wav_arr))
        snr_estimate = 10 * np.log10(signal_power / (noise_power_estimate + 1e-10))

        if debug:
            # 输出特征值，便于调试
            print(f"std: {std_dev:.4f} entropy:{normalized_entropy:.4f} SNR:{snr_estimate:.4f}dB")

        # 条件判断
        if std_dev < threshold_std or normalized_entropy > threshold_entropy:
            return True  # 是纯噪声
        else:
            return False  # 不是纯噪声

    @staticmethod
    def is_abnormal_duration(wav_arr, wav_sr, text, lang, hold1=6, hold2=2, **kwargs):
        lang = lang.lower()
        get_token_num = lambda t, tlang: len(t.split(" ")) if tlang in ["en", "en_us"] else len(t)
        # 根据角色音从 estimate_audio_len.py 里拟合出来的字符数与音频时长公式
        # a, b, c = (0.0004416210417798762, 0.2990780290048899, 0.5942355219347748)
        if lang in ["en", "en_us"]:  # todo only 'en' should be good enough
            # Mike
            # a, b, c = (0.00288095448943406, 0.22635555061873977, 0.7276428136650496)
            # NinaV2
            # a, b, c = (0.006569098898731953, 0.15735181885149516, 0.814867789439186)
            # 基于6个英文角色音的训练样本的ASR结果
            a, b, c, d = (0.0016008349175017078, -0.06599824444946428, 0.9600194411934183, -0.40882416983497394)
        elif lang in ["zh", "zh_cn"]:
            # test_cxm
            # a, b, c = (0.01596191493148245, -0.15270688734650273, 1.7947908880583727)
            # ChatTTS_Voice_Clone_User_3951_20250123010229136_x8bf (cxm)
            a, b, c, d = (1.4589877459468358e-05, -0.003125013569752014, 0.2354471444393487, -1.7437911823924999)
        else:
            logging.warning(f">>> 语言'{lang}' 对应的时长预估参数未就绪，借用中文的")
            a, b, c, d = (1.4589877459468358e-05, -0.003125013569752014, 0.2354471444393487, -1.7437911823924999)
        x = get_token_num(text, lang)
        p = round(a * x ** 3 + b * x ** 2 + c * x + d, 4)
        y = round(wav_arr.shape[0] / wav_sr, 4)
        # 如果文本过长的话，跳过检测，拟合公式不准的
        if x >= 20:
            logging.warning(f">>> 检测到合成长文本(英文20个单词 中文20个字):{x} 跳过时长检测")
            return False, 0, y
        else:
            # 如果实际时长与预估时长差距超过3s，认为是异常音频
            logging.debug(f">>> lang={lang}, text={text}, 文本长度x={x} 预估音频时长y={p} 实际音频时长:{y}")
            # 只拦截过长音频
            is_abnormal = y-p >= hold1 or y/p >= hold2
            return is_abnormal, p, y

    @staticmethod
    def is_empty_noise(wav_arr, var_hold=2000, **kwargs):
        # 如果合成的音频数组方差太小，意味着是空白音或者爆音，最多重试三次，正常方差示例:522218,849305
        return np.var(wav_arr) <= var_hold

    @staticmethod
    def audio_check(wav_arr, wav_sr, text, lang, **kwargs):
        cond1 = GSVModel.is_pure_noise(wav_arr, **kwargs)
        cond2, p, y = GSVModel.is_abnormal_duration(wav_arr, wav_sr, text, lang, **kwargs)
        cond3 = GSVModel.is_empty_noise(wav_arr, **kwargs)
        if cond1:
            logging.warning(f">>> 检测为 is_pure_noise")
        if cond2:
            logging.warning(f">>> 检测为 is_abnormal_duration (预测时长={p} 实际时长={y} lang='{lang}' text='{text}' )")
        if cond3:
            logging.warning(f">>> 检测为 is_empty_noise")
        return any([cond1, cond2, cond3])

    # 检测是否为参考音频泄露
    def is_ref_leakage(self, wav_arr, wav_sr, ref_info: ReferenceInfo):
        target_text = ref_info.text
        tgt_lang = ref_info.lang if ref_info.lang in C.LANG_MAP.values() else C.LANG_MAP[ref_info.lang]
        ref_lang = ref_info.lang if ref_info.lang in C.LANG_MAP.values() else C.LANG_MAP[ref_info.lang]
        if self.ref_info is None:
            # 第一次调用时推理一下参考音频
            logging.warning(f">>> 第一次推理，检测时需要先推理一次参考音频 ")
            self.ref_info = ref_info
            synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                                ref_wav_path=ref_info.audio_fp,
                                                prompt_text=ref_info.text, prompt_language=ref_lang,
                                                top_k=30, top_p=0.99, temperature=0.4, speed=1,
                                                ref_free=False, no_cut=True)
            self.ref_sr, self.ref_wav_int16 = list(synthesis_result)[-1]
        elif any([ref_info.text != self.ref_info.text,
                  ref_info.audio_fp != self.ref_info.audio_fp,
                  ref_info.lang != self.ref_info.lang]):
            # 参考音频和之前记录的有区别,视为第一次调用，重新推理参考音频
            logging.warning(f">>> 参考音频信息变动，检测时需要先推理一次参考音频 ")
            self.ref_info = ref_info
            synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                                ref_wav_path=ref_info.audio_fp,
                                                prompt_text=ref_info.text, prompt_language=ref_lang,
                                                top_k=30, top_p=0.99, temperature=0.4, speed=1,
                                                ref_free=False, no_cut=True)
            self.ref_sr, self.ref_wav_int16 = list(synthesis_result)[-1]

        # ref_wav_d = self.ref_wav_int16.shape[0] / self.ref_sr
        return wav_arr.shape[0] == self.ref_wav_int16.shape[0]

    # no_cut置为True时注意不要开头就打句号
    def predict(self, target_text, target_lang,
                ref_info: ReferenceInfo = None,
                top_k=20, top_p=1.0, temperature=0.99, speed=1,
                ref_free: bool = None, no_cut: bool = False, **kwargs):
        # Synthesize audio
        # target_lang是 en_us/zh_cn/jp_jp/kr_ko
        # tgt_lang是 en/all_zh/all_jp/all_ko
        tgt_lang = target_lang if target_lang in C.LANG_MAP.values() else C.LANG_MAP[target_lang]
        ref_lang = ref_info.lang if ref_info.lang in C.LANG_MAP.values() else C.LANG_MAP[ref_info.lang]
        if ref_free is None:
            if tgt_lang in ["all_zh"]:
                # 中文文本太短（小于8个字符）就强制触发ref_free
                ref_free = len(target_text) <= 8
            elif tgt_lang in ["en"]:
                # 英文太短也是
                ref_free = len(target_text.split(" ")) <= 8
            else:
                # 其他语言（日语韩语）20个字符 不太行，强制ref_free=True得了
                # ref_free = len(target_text) <= 15
                ref_free = True
            if ref_free:
                logging.warning(f"语言{tgt_lang} 合成文本过短，自动触发了强制ref_free (文本: '{target_text}')")
        elif ref_free:
            logging.info(f"ref_free=True 推理采用无参考音频模式")
        synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                            ref_wav_path=ref_info.audio_fp,
                                            prompt_text=ref_info.text, prompt_language=ref_lang,
                                            top_k=top_k, top_p=top_p, temperature=temperature, speed=speed,
                                            ref_free=ref_free, no_cut=no_cut, **kwargs)

        wav_sr, wav_arr_int16 = list(synthesis_result)[-1]
        if self.is_ref_leakage(wav_arr_int16, wav_sr, ref_info):
            logging.error(f"严重问题 检测到参考音频泄露，直接按无参模式推理并返回 文本:{target_text} {target_lang}")
            synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                                ref_wav_path=ref_info.audio_fp,
                                                prompt_text=ref_info.text, prompt_language=ref_lang,
                                                top_k=top_k, top_p=top_p, temperature=temperature, speed=speed,
                                                ref_free=True, no_cut=no_cut, **kwargs)
            wav_sr, wav_arr_int16 = list(synthesis_result)[-1]
            return wav_sr, wav_arr_int16

        cnt, max_cnt = 0, 1  # 如果合成的音频数组方差太小，意味着是空白音或者爆音，最多重试三次，正常方差示例:522218,849305
        while self.audio_check(wav_arr_int16, wav_sr, target_text, target_lang) and cnt <= max_cnt:
            if cnt == max_cnt:
                ref_free = True
                logging.warning(f">>> 检测到 异常音频，将进行第{cnt + 1}/{max_cnt + 1}次重试合成(最后一次 强制ref_free=True)")
            else:
                logging.warning(f">>> 检测到 异常音频，将进行第{cnt + 1}/{max_cnt + 1}次重试合成")
            # 这里把温度调高，增加可能性
            synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                                ref_wav_path=ref_info.audio_fp,
                                                prompt_text=ref_info.text, prompt_language=ref_lang,
                                                top_k=top_k, top_p=top_p, temperature=max(0.8, temperature), speed=speed,
                                                ref_free=ref_free, no_cut=no_cut, **kwargs)
            wav_sr, wav_arr_int16 = list(synthesis_result)[-1]
            logging.warning(f">>> 重新合成后 duration={wav_arr_int16.shape[0]/wav_sr} var={np.var(wav_arr_int16)}")
            cnt += 1
            if self.is_ref_leakage(wav_arr_int16, wav_sr, ref_info):
                logging.error(
                    f">>> 严重问题 检测到参考音频泄露，直接按无参模式推理并返回 文本:{target_text} {target_lang}")
                synthesis_result = self.get_tts_wav(text=target_text, text_language=tgt_lang,
                                                    ref_wav_path=ref_info.audio_fp,
                                                    prompt_text=ref_info.text, prompt_language=ref_lang,
                                                    top_k=top_k, top_p=top_p, temperature=temperature, speed=speed,
                                                    ref_free=True, no_cut=no_cut, **kwargs)
                wav_sr, wav_arr_int16 = list(synthesis_result)[-1]
                return wav_sr, wav_arr_int16
        return wav_sr, wav_arr_int16


def webui_list_audios(directory, text_list=None, cmt_list=None, gr_share=False, port=6006):
    # 检查目录是否存在
    assert os.path.exists(directory), f"指定的目录 {directory} 不存在。"

    # 获取目录下所有的音频文件（假设支持 .wav、.mp3 格式，可按需添加其他格式）
    audio_extensions = ('.wav', '.mp3', 'm4a')
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(audio_extensions)]
    audio_files = sorted(audio_files, key=lambda x: os.path.getmtime(x))
    assert audio_files, f"指定的目录 {directory} 中没有找到音频文件 (.wav .mp3 .m4a)"

    with gr.Blocks() as demo:
        gr.Markdown("### 音频文件试听")
        for idx,fp in enumerate(audio_files):
            with gr.Row():
                with gr.Column():
                    gr.Audio(value=fp, type="filepath", label=os.path.basename(fp))
                with gr.Column():
                    with gr.Row():
                        if text_list is not None:
                            gr.Markdown(text_list[idx])
                    with gr.Row():
                        if cmt_list is not None:
                            gr.Markdown(cmt_list[idx])
        # gr.render()

    if gr_share:
        demo.launch(share=True)
    else:
        demo.launch(server_port=port)
    # demo.close()


# 由于用到了相对路径的import，必须以module形式执行
# python -m service_GSV.GSV_model <sid> <lang>
# -- 这个没就绪 python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_KellyV2 en
# python -m service_GSV.GSV_model test_cxm en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_PhenixV2 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_PhenixV2 en manual
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_OrionV2 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_OrionV2.1 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_MarthaV2 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_ZoeV2 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_Common_NinaV2 en
# python -m service_GSV.GSV_model ChatTTS_Voice_Clone_User_3125_20250307140742211_jwa0 en
if __name__ == '__main__':
    local_test_dir = "audio_test"
    if len(sys.argv) >= 3:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        sid = sys.argv[1]
        lang = sys.argv[2]
        ref = sys.argv[3] if len(sys.argv) >= 4 else C.D_REF_SUFFIX
        print(f">>> Inference with sid={sid} lang={lang} ref={ref}")
        sovits_fp = R.get_sovits_fp(sid)
        gpt_fp = R.get_gpt_fp(sid)
        M = GSVModel(sovits_model_fp=sovits_fp, gpt_model_fp=gpt_fp)
        ref_info = ReferenceInfo.from_sid(sid, suffix=ref)

        # sf.write(os.path.join(opt_dir, "res_audio.wav"), np.hstack(audio_list), sr)
    else:
        # opt_dir = os.path.join(local_test_dir, "cxm_from_webui")
        # os.makedirs(opt_dir, exist_ok=True)
        # # sovits_model = "/root/autodl-fs/models/cxm_from_webui/sovits_xxx_e8_s80.pth"
        # # gpt_model = "/root/autodl-fs/models/cxm_from_webui/gpt_xxx-e15.ckpt"
        # # ref_audio = ReferenceInfo(audio_fp="/root/autodl-fs/models/cxm_from_webui/ref_audio.wav",
        # #                           text="大家好，欢迎来到我的直播间，我是你们的主播小美。",
        # #                           lang="ZH")

        # lang = "en"
        # sid = "lydia"
        # sovits_fp = "/root/GPT-SoVITS/SoVITS_weights_v2/xxx_e8_s168.pth"
        # gpt_fp = "/root/GPT-SoVITS/GPT_weights_v2/xxx-e15.ckpt"
        # M = GSVModel(sovits_model_fp=sovits_fp, gpt_model_fp=gpt_fp)
        # ref_info = ReferenceInfo(
        #     audio_fp="/root/autodl-fs/voice_sample/Lýdia_Machová/denoise_opt/Lýdia_Machová-1.wav_0000000000_0000138240.wav",
        #     text="In fact, I love it so much that I like to learn a new language every two years.",
        #     lang="EN")

        if False: # download from qiniu
            import utils_audio
            sid = "ChatTTS_Voice_Clone_User_3125_20250307140742211_jwa0"

            utils_audio.download_from_qiniu(R.get_ref_audio_osskey(sid), R.get_ref_audio_fp(sid))
            R.get_ref_text_osskey(sid)
            R.get_sovits_osskey(sid)
            R.get_gpt_osskey(sid)
            pass


        lang = "en"
        sid = "amber"
        sovits_fp = "/root/GPT-SoVITS/SoVITS_weights_v2/xxx_e12_s252.pth"
        gpt_fp = "/root/GPT-SoVITS/GPT_weights_v2/xxx-e15.ckpt"
        ref_audio_fp = "/root/autodl-fs/voice_sample/amber/denoise_opt/amber_20.wav"
        M = GSVModel(sovits_model_fp=sovits_fp, gpt_model_fp=gpt_fp)
        ref_info = ReferenceInfo(
            audio_fp=ref_audio_fp,
            text="Are you okay? Go ask the boss later to get out some money to switch to 714.",
            lang="EN")


        if False:  # post2oss
            import utils_audio
            import subprocess
            url = utils_audio.post2qiniu(sovits_fp, R.get_sovits_osskey(sid))
            logging.info(f">>> url as: '{url}'")
            logging.info(">>> Uploading gpt model to qiniu.")
            url = utils_audio.post2qiniu(gpt_fp, R.get_gpt_osskey(sid))
            logging.info(f">>> url as: '{url}'")
            logging.info(">>> Uploading default_ref to qiniu.")
            res = subprocess.run(["cp", "/root/autodl-fs/voice_sample/Lýdia_Machová/denoise_opt/Lýdia_Machová-1.wav_0000000000_0000138240.wav",
                                  R.get_ref_audio_fp(sid, C.D_REF_SUFFIX)],
                                 capture_output=True, text=True, encoding='utf-8')
            assert res.returncode == 0, res.stdout.strip()
            res = subprocess.run(["echo", "\"EN|In fact, I love it so much that I like to learn a new language every two years.\"",
                                  ">",
                                  R.get_ref_text_fp(sid, C.D_REF_SUFFIX)],
                                 capture_output=True, text=True, encoding='utf-8')
            assert res.returncode == 0, res.stdout.strip()
            audio_url = utils_audio.post2qiniu(R.get_ref_audio_fp(sid, C.D_REF_SUFFIX), R.get_ref_audio_osskey(sid))
            text_url = utils_audio.post2qiniu(R.get_ref_text_fp(sid, C.D_REF_SUFFIX), R.get_ref_text_osskey(sid))
            logging.info(f">>> [audio_url]:'{audio_url}' [text_url]:'{text_url}'")

    opt_dir = os.path.join(local_test_dir, sid)
    if os.path.exists(opt_dir):
        shutil.rmtree(opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    audio_list = []
    cmt_list = []
    text_fp = "./core/quality_check/text_en_us.txt"
    with open(text_fp, "r", encoding="utf-8") as fr:
        lines = [i.strip() for i in fr.readlines()]
        lines = [i for i in lines if len(i) >= 1]
    for idx, line in enumerate(lines):
        begin_t = time.time()
        logging.info(f">>> 开始合成({idx}/{len(lines)}): {line}")
        sr, audio = M.predict(target_text=line,
                              target_lang=lang,
                              ref_info=ref_info,
                              top_k=30, top_p=0.99, temperature=0.6,
                              no_cut=True)
        end_t = time.time()
        audio_list.append(audio)
        opt_fp = f"audio{idx}_{len(line.split(' '))}token_{(end_t - begin_t) * 1000:.0f}ms.wav"
        is_leak = M.is_ref_leakage(audio, sr, ref_info)
        is_any_abnormal = M.audio_check(wav_arr=audio, wav_sr=sr, text=line, lang=lang)
        opt_fp = os.path.join(opt_dir, opt_fp)
        sf.write(opt_fp, audio, sr)
        cmt_list.append(
            f"单词数: {len(line.split(' '))} 合成耗时: {(end_t - begin_t) * 1000:.0f}ms 异常检测: leak={is_leak} abnormal={is_any_abnormal}")
        # from pydub import AudioSegment
        # audio_segment = AudioSegment(
        #     audio.tobytes(),
        #     frame_rate=sr,
        #     sample_width=audio.dtype.itemsize,
        #     channels=1
        # )
        # audio_segment.export(opt_fp, format='m4a')
    webui_list_audios(opt_dir, text_list=lines, cmt_list=cmt_list)

