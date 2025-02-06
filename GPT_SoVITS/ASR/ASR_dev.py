from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", model_revision="v2.0.4")

import soundfile
import os

audio_dir = "/Users/bytedance/AudioProject/GPT-SoVITS/GPT_SoVITS/tmp_output"
fp_list = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) if i.endswith("wav")]
cache, total_num = {}, len(fp_list)
for idx, fp in enumerate(fp_list):
    wav, sr = soundfile.read(fp)
    res = model.generate(input=wav, cache=cache,
                         is_final=True if idx == total_num-1 else False, chunk_size=chunk_size,
                         encoder_chunk_look_back=encoder_chunk_look_back,
                         decoder_chunk_look_back=decoder_chunk_look_back)
    print(res[0])

#
# wav_file = os.path.join(model.model_path, "/Users/bytedance/Downloads/c09b10ad-db0e-4c9e-919c-969dc3b727e8.wav")
# speech, sample_rate = soundfile.read(wav_file)
# chunk_stride = chunk_size[1] * 960  # 600ms
#
# cache = {}
# total_chunk_num = int(len(speech-1)/chunk_stride+1)
# for i in range(total_chunk_num):
#     speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
#     is_final = i == total_chunk_num - 1
#     res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#     print(res, cache['prev_samples'])


# frontend: (['reserve_waveforms', 'input_cache', 'lfr_splice_cache', 'waveforms', 'fbanks', 'fbanks_lens']
# frontend: (['reserve_waveforms', 'input_cache', 'lfr_splice_cache', 'waveforms', 'fbanks', 'fbanks_lens']
