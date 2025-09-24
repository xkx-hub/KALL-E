
import os
import yaml
import json
import torch
import librosa
import argparse
import numpy as np
import torch.distributions as D

from tqdm import tqdm
from model import Kalle
from flowvae import FlowVAE as Generator
from transformers import AutoTokenizer,get_scheduler
from scipy.io.wavfile import write


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class infer_tools:
    def __init__(self, 
                 device = 0, 
                 tokenizer_path = './tokenizer_dir',
                 vae_config = 'configs/flowvae_dim1024_12p5hz.json',
                 vae_model_path = 'ckpt/flowvae.pt',
                 check_point_path = 'ckpt/model.pt',
                 ):
        self.device = f"cuda:{device}"
        self.tokenizer_path = tokenizer_path
        self.vae_config = vae_config
        self.vae_model_path = vae_model_path
        self.check_point_path = check_point_path

        self.init_vae_generator()
        self.init_tokenizer()
        self.init_model()
        
    def init_vae_generator(self):
        with open(self.vae_config) as f:
            vae_model_config = f.read()

        json_config = json.loads(vae_model_config)
        self.h = AttrDict(json_config)
        generator = Generator(self.h)

        state_dict_g = torch.load(self.vae_model_path, map_location='cpu')
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        generator.to(self.device)
        torch.backends.cudnn.benchmark = False
        self.generator = generator

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')     

        self.tokenizer = tokenizer

    def init_model(self):
        model = Kalle()
        self.ckpt = torch.load(self.check_point_path,map_location='cpu')
        model.load_state_dict(self.ckpt)
        # for name, param in model.named_parameters():
        #     assert param.dtype == torch.float32
        self.model = model.to(self.device)

    def reinit_model(self):
        self.model.load_state_dict(self.ckpt)
        self.model = self.model.to(self.device)


    def infer(self,
              target_text,
              prompt_text = None,
              prompt_wav_path = None,):
        
        if prompt_wav_path is not None or prompt_text is not None:
            assert prompt_wav_path is not None and prompt_text is not None, "prompt_wav_path and prompt_text must be set together"

            prompt_text_tokenized = self.tokenizer.encode(prompt_text)
            prompt_wav,_ = librosa.load(prompt_wav_path, sr=self.h.sampling_rate, mono=True)
            prompt_wav_tensor = torch.FloatTensor(prompt_wav.reshape(1, -1)).to(self.device)
            with torch.no_grad():
                prompt_mean_scale_latent = self.generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
            mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
            audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean

            target_text_tokenized = self.tokenizer.encode(target_text)[1:]
        else:
            prompt_text_tokenized = []
            target_text_tokenized = self.tokenizer.encode(target_text)
            audio_latents = None
        
        text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        input_ids = text_ids.to(self.device)


        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                generate_audio_latents = self.model.infer(input_ids,audio_latents)
                audio = self.generator.inference_from_latents(generate_audio_latents, do_sample=True) * 32767.0

        audio = audio.detach().cpu().numpy().astype('int16')
        write("output.wav",self.h.sampling_rate,audio)
        return audio



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_text', type=str, default="Hello, have a nice day!", help='Target text path')
    parser.add_argument('--prompt_text', type=str, default=None, help='Prompt text path')
    parser.add_argument('--prompt_wav_path', type=str, default=None, help='Prompt wav path')
    
    args = parser.parse_args()

    infer = infer_tools()
    infer.infer(target_text=args.target_text,prompt_text=args.prompt_text,prompt_wav_path=args.prompt_wav_path)