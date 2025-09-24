import torch,random
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from ecapa_tdnn import ECAPA_TDNN
from transformers import LlamaConfig,AutoModelForCausalLM

default_llama_config = LlamaConfig(
    attention_dropout=0.0,
    attention_bias=False,
    bos_token_id=128000,
    eos_token_id=[128001, 128008, 128009],
    head_dim=64,
    hidden_act="silu",
    hidden_size=2048,
    intermediate_size=8192,
    max_position_embeddings=131072,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=32,
    num_hidden_layers=16,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
    rope_theta=500000.0,
    tie_word_embeddings=True,
    torch_dtype=torch.float16,
    vocab_size=128264,
    use_cache=True
)

class Kalle(nn.Module):
    def __init__(
        self,
        config = default_llama_config,
        use_flash_attention = False
    ):
        super().__init__()

        self.use_fa = use_flash_attention
        if self.use_fa:
            self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_config(
                config
            )

        self.audio_linear = nn.Linear(512,2048,dtype=torch.bfloat16) if self.use_fa else nn.Linear(512,2048)
        self.distribution_linear = nn.Linear(2048,1024,dtype=torch.bfloat16) if self.use_fa else nn.Linear(2048,1024)

        # not for the pretain model
        # self.speaker_encoder = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=2048)
        # self.speaker_cond_disp_linear = nn.Linear(2048,2048*2)



    @torch.no_grad()
    def infer(
        self,
        input_ids,             
        audio_latents,          
        end_disp_kl_thres = 0.5,
        max_length = 1000,

    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) if audio_latents is not None else None

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        final_audio_latents_lst = []

        for i in range(max_length):
            hidden = self.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = self.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)

            audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

            if kl < end_disp_kl_thres and i > 3:
                break

            audio_embed = self.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)

        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)
