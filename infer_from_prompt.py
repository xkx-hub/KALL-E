import torch
import json
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoTokenizer
from model import Kalle
from wav_decoder import AttrDict
from wav_decoder import BigVGANDecoder as WavDecoder

model = Kalle()
ckpt = torch.load('/home/work_nfs16/kxxia/work/epoch_3_step_650155.pt',map_location='cpu')
model.load_state_dict(ckpt)
model = model.to('cuda')

wav_decoder_config = open("./decoder_config.json").read()
h = AttrDict(json.loads(wav_decoder_config))
wavdecoder = WavDecoder(h)
state_dict_g = torch.load('./decoder.pt', map_location='cpu')
wavdecoder.load_state_dict(state_dict_g)
wavdecoder.eval()
wavdecoder.remove_weight_norm()
wavdecoder.to('cuda')
torch.backends.cudnn.benchmark = False


tokenizer = AutoTokenizer.from_pretrained('/home/work_nfs16/kxxia/work/acc_llasa/tokenizer_dir')
speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    
speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')     

prompt_text_tokenized = []
target_text_tokenized = tokenizer.encode("It was, however, still so hot from its flight through the air.")
audio_latents = None

text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized + [128263, 128260] )).long()
input_ids = text_ids.to('cuda')

with torch.no_grad():
    with torch.autocast(device_type="cuda"):
        generate_audio_latents = model.infer(input_ids,audio_latents)
        audio = wavdecoder.inference_from_latents(generate_audio_latents, do_sample=True) * 32767.0

audio = audio.detach().cpu().numpy().astype('int16')
write('hello.wav',16000,audio)



