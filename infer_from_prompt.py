import torch
import json
import os
import numpy as np
import argparse
from scipy.io.wavfile import write
from transformers import AutoTokenizer
from model import Kalle
from wav_decoder import AttrDict
from wav_decoder import BigVGANDecoder as WavDecoder

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s','--speaker',
    type=str,
    choices=['70', '159', 'aishell3-SSB0341', 'didispeech-00010111'],
    default='',
    help='Target speaker (70, 159, aishell3-SSB0341, didispeech-00010111)'
)
parser.add_argument('-t','--text', type=str, default="你好，今天天气很好。", help='Text to synthesize')
parser.add_argument('-m','--model', type=str, default="./checkpoints/kalle_model.pt", help='Model file')
parser.add_argument('-o','--output', type=str, default="output.wav", help='Output file')

args = parser.parse_args()

prompt_latent_path = os.path.join('prompt_dir',args.speaker + '.npy')
prompt_text_path = os.path.join('prompt_dir',args.speaker + '.txt')
if os.path.exists(prompt_latent_path) and os.path.exists(prompt_text_path):
    prompt_latent = torch.from_numpy(np.load(prompt_latent_path))
    with open(prompt_text_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
else:
    prompt_latent = None
    prompt_text = None

target_text = args.text

############################ init model ############################
model = Kalle()
# ckpt = torch.load('/home/work_nfs16/kxxia/work/epoch_3_step_650155.pt',map_location='cpu')
ckpt = torch.load(args.model,map_location='cpu')
model.load_state_dict(ckpt)
model = model.to('cuda')



############################ init wav decoder ############################
wav_decoder_config = open("./decoder_config.json").read()
h = AttrDict(json.loads(wav_decoder_config))
wavdecoder = WavDecoder(h)
state_dict_g = torch.load('./checkpoints/decoder.pt', map_location='cpu')
wavdecoder.load_state_dict(state_dict_g)
wavdecoder.eval()
wavdecoder.remove_weight_norm()
wavdecoder.to('cuda')
torch.backends.cudnn.benchmark = False


tokenizer = AutoTokenizer.from_pretrained('./tokenizer_dir')
speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    
speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')     

prompt_text_tokenized = tokenizer.encode(prompt_text) if prompt_text else []
target_text_tokenized = tokenizer.encode(target_text)
if len(prompt_text_tokenized) > 0:
    target_text_tokenized = target_text_tokenized[1:]


text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized + [speech_understanding_end_id, speech_generation_start_id ] )).long()
input_ids = text_ids.to('cuda')

if prompt_latent is not None:
    mean, logs_scale = prompt_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
    audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean
else:
    audio_latents = None


with torch.no_grad():
    with torch.autocast(device_type="cuda"):
        generate_audio_latents = model.infer(input_ids,audio_latents)
        audio = wavdecoder.inference_from_latents(generate_audio_latents, do_sample=True) * 32767.0

audio = audio.detach().cpu().numpy().astype('int16')
write(args.output,16000,audio)



