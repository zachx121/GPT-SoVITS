import whisper
import sys
import numpy as np
import ffmpeg
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as write_wav
import os
import torch
import logging
from typing import Union

logging.basicConfig(format='[%(asctime)s-%(levelname)s]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from ASR_WhisperCustomModel import ModelDimensions, Whisper


class Speech2Text:
    def __init__(self, model_type, download_root="./whisper_models", device=None):
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device if device is not None else default_device
        self.model_type = model_type
        self.download_root = download_root
        self.model: Whisper = None
        self.is_init = False
        # 测试了下3090上3个medium进程并行在(5,2)配置上也达不到准实时
        # 测试了下3090上3个base进程并行在(5,5)配置上可以达到准实时
        self.beam_size = 5 if self.model_type == "base" else 3
        self.best_of = 2 if self.model_type == "base" else 1

    def init(self, prehot_audio="./prehot_speech2text.wav"):
        # logging.info(">>> loading whiser model")
        # self.model = whisper.load_model(self.model_type, download_root=self.download_root)
        self.model: Whisper = self.__load_model()
        # logging.info(">>> loading whiser model(done.)")
        try:
            logging.info(">>> try to transcribe with prehot_audio")
            logging.debug("    result: '%s'" % self.transcribe(prehot_audio, fp16=False))
        except (Exception,) as e:
            logging.warning("pre_hot fail. %s" % e)
        self.is_init = True
        # self.model.share_memory()  # 不支持sparseTensor
        return self

    def __load_model(self) -> Whisper:
        _ALIGNMENT_HEADS = {
            "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
            "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
            "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
            "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
            "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
            "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
            "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
            "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
            "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
            "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
            "large": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        }
        alignment_heads = _ALIGNMENT_HEADS[self.model_type]
        checkpoint_file = os.path.join(self.download_root, self.model_type+".pt")
        logging.debug("loading from ckpt_file: %s" % checkpoint_file)
        open(checkpoint_file, "rb")
        with open(checkpoint_file, "rb") as fp:
            checkpoint = torch.load(fp, map_location=self.device)
        # logging.debug("loading from ckpt_file: %s (done.)" % checkpoint_file)
        del checkpoint_file

        #logging.debug("executing ModelDimensions")
        dims = ModelDimensions(**checkpoint["dims"])
        #logging.debug("executing Whisper")
        # logging.debug(">>> ModelDimensions as follow:")
        # logging.debug(str(dims))
        # logging.debug(">>>")
        model = Whisper(dims)
        #logging.debug("executing load_state_dict")
        model.load_state_dict(checkpoint["model_state_dict"])

        if alignment_heads is not None:
            # logging.debug("use alignment_heads as: %s" % alignment_heads)
            model.set_alignment_heads(alignment_heads)

        #logging.debug("executing model.to(self.device)")
        return model.to(self.device)

    # prob_holder: 0.5表示只有当no_speech的概率低于0.5的时候才返回转录的文字，否则给空串
    def transcribe(self, audio_inp: Union[str, np.ndarray, torch.Tensor],
                   prob_holder=0.9,
                   return_details=False,
                   **kwargs):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        result = self.model.transcribe(audio_inp,
                                       task="transcribe",
                                       beam_size=self.beam_size,
                                       best_of=self.best_of,
                                       word_timestamps=False,
                                       **kwargs)

        if return_details:
            return "".join(["%s_%.4f " % (seg["text"], seg['no_speech_prob']) for seg in result['segments']])
        elif prob_holder == 1.0:
            res = "".join([seg["text"] for seg in result['segments']])
            return res
        else:
            res = "".join([seg["text"] for seg in result['segments'] if float(seg['no_speech_prob']) <= prob_holder])
            res = "[EMPTY]" if res == "" else res
            return res

    def transcribe_buffer(self, audio_buffer,
                          sr_inp,
                          channels_inp,
                          return_details=False,
                          prob_holder=0.9, **kwargs):
        assert self.model is not None, "self.model is None, should call '.init()' at first"
        audio = self.load_audio_raw(audio_buffer, sr_inp=sr_inp, channels=channels_inp)
        # logging.debug("transcribe_buffer>load_audio_raw done.(%s)" % audio.shape)
        result = self.model.transcribe(audio,
                                       task="transcribe",
                                       beam_size=self.beam_size,
                                       best_of=self.best_of,
                                       word_timestamps=False,
                                       **kwargs)
        if return_details:
            return "".join(["%s_%.4f " % (seg["text"], seg['no_speech_prob']) for seg in result['segments']])
        elif prob_holder == 1.0:
            # 默认是不做任何分数控制
            res = "".join([seg["text"] for seg in result['segments']])
            return res
        else:
            res = "".join([seg["text"] for seg in result['segments']
                           if float(seg['no_speech_prob']) <= kwargs.get("prob_holder", 1.0)])
            return res

    @staticmethod
    def load_audio_raw(pcm_data: bytes,
                       sr_inp: int = 44100,
                       channels: int = 2,
                       sample_fmt: str = 's16le',
                       sr_out: int = 16000):
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input('pipe:', format=sample_fmt, acodec='pcm_s16le', ac=channels, ar=sr_inp)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr_out)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=pcm_data)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: (str, bytes), sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


load_audio_raw = Speech2Text.load_audio_raw


if __name__ == '__main__':
    if False:
        import whisper.tokenizer
        print("all language: %s" % whisper.tokenizer.LANGUAGES.keys())
    # process file.
    if True:
        FILE_FP = sys.argv[1] if len(sys.argv) >= 2 else "./prehot_speech2text.wav"
        MODEL_TYPE = sys.argv[2] if len(sys.argv) >= 3 else "tiny"
        MODEL_DIR = sys.argv[3] if len(sys.argv) >= 4 else "/root/autodl-fs/whisper"
        logging.info(">>> use FILE_FP as '%s'" % FILE_FP)
        logging.info(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
        M_stt = Speech2Text(model_type=MODEL_TYPE, download_root=MODEL_DIR).init()
        text = M_stt.transcribe(FILE_FP, fp16=False)
        logging.info(">>> transcribe_file:\n%s" % text)
        text = M_stt.transcribe("prehot_speech2text.wav", fp16=False, language="zh")
        logging.info(">>> transcribe_file(lang=zh):\n%s" % text)
        # text = M_stt.transcribe(FILE_FP, fp16=False, language=None)
        # logging.info(">>> transcribe_file(lang=en):\n%s" % text)

    if False:
        MODEL_TYPE = "tiny"
        logging.info(">>> use MODEL_TYPE as '%s'" % MODEL_TYPE)
        M_stt = Speech2Text(model_type=MODEL_TYPE, download_root="./whisper_models").init()
        rate, raw_audio = wav_read('/Users/didi/0-Code/samples/CXM/audio_daniel_2021-part0.wav')
        if raw_audio.dtype == np.int16:
            sampwidth = 2
        elif raw_audio.dtype == np.int32:
            sampwidth = 4
        else:
            raise Exception("unexpected type: %s" % raw_audio.dtype)
        slice_length = 3 * rate
        for i in range(0, len(raw_audio), slice_length):
            audio_slice = raw_audio[i:i + slice_length]
            data = {"audio": audio_slice.tobytes(),
                    "audio_shape": audio_slice.shape,
                    "sample_rate": rate}
            text = M_stt.transcribe_buffer(data['audio'],
                                           sr_inp=data['sample_rate'],
                                           channels_inp=data['audio_shape'][1],
                                           fp16=False)
            print(text)

        sys.exit(0)
        audio_slice = raw_audio[0:0 + slice_length]
        data = {"audio": audio_slice.tobytes(), "audio_shape": audio_slice.shape, "sample_rate": rate}
        # MOCK
        audio_buffer = data['audio']
        audio_shape = data['audio_shape']
        audio_sr = data['sample_rate']
        audio_raw = np.frombuffer(audio_buffer, np.int16).flatten().astype(np.float32) * (1.0 / 2 ** 15)

        # audio_slice: (132300, 2) int16, scipy.io.wavfile直接读出来的数组
        # audio_buffer: b'\x00....' bytes, 数组直接audio_slice.tobytes()的结果
        # audio_raw: (264600,) float32, 字节流转回数组并通过除以2^15次归一化
        #   - audio_raw.reshape(audio_shape)*2**15 就变回 audio_slice了
        #
        # load_audio("./tmp2.wav"): (48000,) float32,
        #   - 文件由audio_raw使用scipy.io.wavfile.write，reshape(-1,2)后写入
        #   - 原始数组audio_raw的采样率是 44100
        #   - 函数load_audio默认采样率是 16000
        #   - e.g. 44100/16000==audio_raw.reshape(-1,2).shape[0]/load_audio("./tmp2.wav").shape[0]
        # load_audio(audio_buffer): (154345,) float32,

        # get correct result | re-construct & write to file.
        write_wav("./tmp2.wav", audio_sr, audio_raw.reshape(audio_shape))
        print("from .wav: " + M_stt.transcribe("./tmp2.wav", fp16=False))
        # get correct result | load_audio with the file.
        print("from load_audio(.wav): " + M_stt.transcribe(load_audio("./tmp2.wav"), fp16=False))
        # get empty string
        print(load_audio("./tmp2.wav").shape)
        print(load_audio_raw(audio_buffer,sr=audio_sr,channels=2).shape)
        print("from load_audio_raw(buffer): " + M_stt.transcribe(load_audio_raw(audio_buffer, sr=audio_sr, channels=2), fp16=False))
        print("from load_audio(buffer): " + M_stt.transcribe(load_audio(audio_buffer), fp16=False))
        print("from audio_raw: " + M_stt.transcribe(audio_raw, fp16=False))



        #text = M_stt.transcribe_data(audio, audio_sr)
    # timeit
    # import timeit
    # time_res = timeit.timeit(lambda: M_stt.transcribe("./audio_daniel_2021-part0.wav"), number=3)
    # logging.info(">>> timeit: %.4f" % (time_res/3.0))

    # import pickle
    # with open("./result_%s.pkl" % MODEL_TYPE, "wb") as f:
    #     pickle.dump(result, f)

    # import json
    # print(json.dumps(result))
