
import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

import os
import soundfile as sf
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

GPT_MODEL_FP = "./pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SOVITS_MODEL_FP = "./pretrained_models/s2G488k.pth"
LANG_MAP = {"JP": "Japanese", "ZH": "Chinese", "EN": "English"}
TEMP_PATH = "./tmp_output"
os.makedirs(TEMP_PATH, exist_ok=True)


class ReferenceInfo:
    def __init__(self, audio_fp: str, text: str, lang: str):
        self.audio_fp = audio_fp
        self.text = text
        self.lang = lang


class GSVModel:
    def __init__(self, gpt_model_fp=None, sovits_model_fp=None):
        self.gpt_model_fp = GPT_MODEL_FP if gpt_model_fp is None else gpt_model_fp
        self.sovits_model_fp = SOVITS_MODEL_FP if sovits_model_fp is None else sovits_model_fp
        logging.info(f">>> Init GSVModel with:")
        logging.info(f"    [gpt_model_fp]: '{self.gpt_model_fp}'")
        logging.info(f"    [sovits_model_fp]: '{self.sovits_model_fp}'")

    def predict(self, target_text, target_lang,
                ref_info: ReferenceInfo = None,
                output_path: str = TEMP_PATH):
        # Change model weights
        change_gpt_weights(gpt_path=self.gpt_model_fp)
        change_sovits_weights(sovits_path=self.sovits_model_fp)

        # Synthesize audio
        synthesis_result = get_tts_wav(ref_wav_path=ref_info.audio_fp,
                                       prompt_text=ref_info.text,
                                       prompt_language=LANG_MAP[ref_info.lang],
                                       text=target_text,
                                       text_language=LANG_MAP[target_lang], top_p=1, temperature=1)

        result_list = list(synthesis_result)

        if result_list:
            last_sampling_rate, last_audio_data = result_list[-1]
            output_wav_path = os.path.join(output_path, "output.wav")
            sf.write(output_wav_path, last_audio_data, last_sampling_rate)
            logging.info(f"Audio saved to {output_wav_path}")


if __name__ == '__main__':
    M = GSVModel(gpt_model_fp="/Users/bytedance/AudioProject/GPT-SoVITS/GPT_weights/XuRan-e15.ckpt",
                 sovits_model_fp="/Users/bytedance/AudioProject/GPT-SoVITS/SoVITS_weights/XuRan_e8_s96.pth")
    M.predict(target_text="您好，我是您的个人助理！您可以说今天天气如何。",
              target_lang="ZH",
              ref_info=ReferenceInfo(audio_fp="/Users/bytedance/AudioProject/voice_sample/Xu_Ran/vocals copy 7.wav",
                                     text="我想问一下，就是咱们那个疫情防控政策",
                                     lang="ZH"))
