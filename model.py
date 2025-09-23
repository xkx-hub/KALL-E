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


# class Kalle(nn.Module):
#     def __init__(
#         self,
#         config = default_llama_config,
#         use_flash_attention = True
#     ):
#         super().__init__()

#         self.use_fa = use_flash_attention
#         if self.use_fa:
#             self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_config(
#                 config,
#                 attn_implementation="flash_attention_2",
#                 torch_dtype=torch.bfloat16,
#             )
#         else:
#             self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_config(
#                 config
#             )

#         self.audio_linear = nn.Linear(512,2048,dtype=torch.bfloat16) if self.use_fa else nn.Linear(512,2048)
#         self.distribution_linear = nn.Linear(2048,1024,dtype=torch.bfloat16) if self.use_fa else nn.Linear(2048,1024)

#         self.speaker_encoder = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=2048)
#         self.speaker_cond_disp_linear = nn.Linear(2048,2048*2)

#     def forward(
#         self,
#         input_ids,              # b,t
#         audio_latents,          # b,t,d1
#         audio_distribution_l,     # b,t,d2
#         mels,                   # b,d,t

#         ids_mask,
#         audio_mask,
#         target_mask,
#         end_mask,
#     ):
#         text_embed = self.base_model.model.embed_tokens(input_ids)  # b,t,d
#         audio_embed = self.audio_linear(audio_latents)              # b,t,d
#         audio_latents_dim = audio_latents.shape[-1]

#         speaker_cond = self.speaker_encoder(mels.transpose(1,2))
#         speaker_cond_disp = self.speaker_cond_disp_linear(speaker_cond)
#         speaker_cond_mean,speaker_cond_logs_scale = speaker_cond_disp.chunk(2,dim=1)
#         speaker_cond = torch.randn_like(speaker_cond_mean) * torch.exp(speaker_cond_logs_scale) + speaker_cond_mean

#         norm_disp = D.Normal(torch.zeros_like(speaker_cond_mean),torch.ones_like(speaker_cond_logs_scale))
#         speaker_disp = D.Normal(speaker_cond_mean,  torch.exp(speaker_cond_logs_scale))
#         speaker_cond_kl = D.kl_divergence(speaker_disp ,norm_disp).sum(1) / 2048
#         speaker_cond_kl = speaker_cond_kl.sum() / speaker_cond_kl.shape[0]

#         input_embed = (audio_embed * audio_mask.unsqueeze(-1)) + (text_embed * ids_mask.unsqueeze(-1))   # b,t,d
#         attention_mask = ids_mask + audio_mask
#         input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)

#         true_column = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
#         attention_mask = torch.cat((true_column,attention_mask ), dim=1)

#         hidden = self.base_model.model(
#             inputs_embeds=input_embed,
#             attention_mask=attention_mask,
#         )[0]  # b,t,d
#         hidden = hidden[:,1:,:]

#         distribution_p = self.distribution_linear(hidden)       # b,t,d2

#         mean1,logs_scale1 = audio_distribution_l.chunk(2,dim=2)
#         mean2,logs_scale2 = distribution_p.chunk(2,dim=2)

#         std1 = torch.exp(logs_scale1)
#         std2 = torch.exp(logs_scale2)

#         l_disp = D.Normal(mean1,std1) # 
#         p_disp = D.Normal(mean2,std2)

#         kl = D.kl_divergence(p_disp, l_disp)

#         kl = kl.sum(2) / audio_latents_dim
#         audio_loss = (kl * target_mask).sum() / target_mask.sum()
#         end_loss = (kl * end_mask).sum() / end_mask.sum()

#         return {
#             "speaker_cond_kl": speaker_cond_kl,
#             "audio_loss": audio_loss,
#             "end_loss": end_loss,
#             "pre_mean": mean2,
#             "pre_log_scale": logs_scale2
#         }

#     @torch.no_grad()
#     def infer(
#         self,
#         input_ids,              # t
#         audio_latents,          # 
#         mels,                   # b,d,t
#         end_disp_kl_thres = 0.5,
#         max_length = 1000,

#     ):
#         text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
#         audio_embed = self.audio_linear(audio_latents) if audio_latents is not None else None

#         self.speaker_encoder.eval()
#         if mels is not None :
#             speaker_cond = self.speaker_encoder(mels.transpose(1,2))    # 1 1024
#             speaker_cond_disp = self.speaker_cond_disp_linear(speaker_cond)
#             speaker_cond_mean,speaker_cond_logs_scale = speaker_cond_disp.chunk(2,dim=1)
#             speaker_cond = torch.randn_like(speaker_cond_mean) * torch.exp(speaker_cond_logs_scale) + speaker_cond_mean

#         else:
#             zero_mean = torch.zeros((1,2048),dtype=text_embed.dtype,device=text_embed.device)
#             one_scale = torch.ones((1,2048),dtype=text_embed.dtype,device=text_embed.device)
#             speaker_cond = torch.randn_like(zero_mean) * one_scale + zero_mean
#             self.speaker_cond = speaker_cond

#         input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
#         input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)
#         final_audio_latents_lst = []

#         for i in range(max_length):
#             hidden = self.base_model.model(inputs_embeds=input_embed)
#             last_hidden = hidden[0][:,-1:,:]
#             last_disp = self.distribution_linear(last_hidden)

#             mean,logs_scale2 = last_disp.chunk(2,dim=2)

#             audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

#             final_audio_latents_lst.append(last_disp)

#             end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
#             p_disp = D.Normal(mean,torch.exp(logs_scale2))
#             latent_dim = mean.shape[2] 
#             kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

#             if kl < end_disp_kl_thres and i > 3:
#                 break

#             audio_embed = self.audio_linear(audio_latent)
#             input_embed = torch.cat((input_embed,audio_embed),dim=1)
            
#         generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
#         return generate_audio_latents.transpose(1,2)
