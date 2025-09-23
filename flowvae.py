import torch
from torch import nn, sin, pow
from torch.nn import Parameter
from torch.nn import functional as F
from alias_free_torch import *

from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Conv1d_S(nn.Module):

    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(self,
           in_channels,
           out_channels,
           kernel_size=1,
           stride=1,
           dilation=1,
           groups=1,
           norm_type="weight_norm",
           init_type=None):

        super(Conv1d_S, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        pad = dilation * (kernel_size - 1) // 2

        self.layer = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
        if init_type == "orthogonal":
            nn.init.orthogonal_(self.layer.weight)
        elif init_type == "normal":
            nn.init.normal_(self.layer.weight, mean=0.0, std=0.01)

        if norm_type == "weight_norm":
            self.layer = weight_norm(self.layer)
        elif norm_type == "spectral_norm":
            self.layer = spectral_norm(self.layer)

    def forward(self, inputs):
        return self.layer(inputs)

class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4):
        super(ResStack, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=base**i, padding=base**i)),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv1d(channel, channel,
                    kernel_size=kernel_size, dilation=1, padding=1)),
            )
            for i in range(nums)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None, causal=True):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], causal=causal))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = 'zeros',
        causal: bool = False,
        input_transpose: bool = False,
        **kwargs
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, \
                    "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )

        self.causal = causal
        self.stride = stride
        self.transpose = input_transpose

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x.transpose(1, 2) if self.transpose else x

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(ConvTranspose1d, self).extra_repr(), self.causal, self.transpose)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=100,
                 base_channels=12,
                 proj_kernel_size=3,
                 stack_kernel_size=3,
                 stack_dilation_base=2,
                 stacks=6,
                 channels=[12, 24, 48, 96, 192, 384, 768],
                 down_sample_factors=[2, 2, 2, 2, 4, 4],
                 use_vae=False):
        super(Encoder, self).__init__()

        act_slope = 0.2
        if use_vae:
            out_channels = out_channels * 2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(in_channels,
                   base_channels,
                   kernel_size=proj_kernel_size,
                   stride=1),
            nn.LeakyReLU(act_slope, True)
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        for (in_c, out_c), down_f in zip(
                zip(channels[:-1], channels[1:]), down_sample_factors):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(act_slope, True)
            ]

        # post layers
        layers += [
            Conv1d_S(channels[-1],
                   out_channels,
                   proj_kernel_size,
                   stride=1),
            #nn.Tanh() TODO
        ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, z_inputs=None):
        return self.generator(conditions)


class FlowVAE(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super(FlowVAE, self).__init__()
        self.h = h
        causal = h.causal

        self.audio_encoder = Encoder(
                out_channels=h.latent_dim, 
                use_vae=h.use_vae,
                channels=h.downsample_channels,
                down_sample_factors=h.downsample_rates)
        
        self.flow = ResidualCouplingBlock(
                h.latent_dim, h.flow_hidden_channels, 5, 1, 4, gin_channels=0, causal=causal)


        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        # I add delay in the first conv
        self.conv_pre = weight_norm(Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, causal=causal))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation, causal=causal))

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):

        x = self.audio_encoder(x)
        assert self.h.use_vae

        m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        
        # Flow
        mask = torch.ones([z.size(0), 1, z.size(-1)]).to(z.device)
        z_p = self.flow(z, mask)

        # pre conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        if self.h.use_vae:
            return x, (z_p, logs_q, None, None)
        else:
            return x, (None, None, None, None)

    def extract_latents(self, x):
        x = self.audio_encoder(x)
        return x

    def inference_from_latents(self, x, do_sample=True):
        if self.h.use_vae and do_sample:
            assert x.size(1) == self.h.latent_dim * 2, "Input must be like [B, D, H]"
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
            m_p = torch.zeros_like(m_q)
            logs_p = torch.ones_like(logs_q)
        else:
            assert x.size(1) == self.h.latent_dim, "Input must be like [B, D, H]"

        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
 

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)



##############################  ↓ stable audio    ########################################
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        bn: bool = False,
        activation = None,
        w_init_gain = None,
        input_transpose: bool = False,
        **kwargs
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)

        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias
        )

        self.in_channels = in_channels
        self.transpose = input_transpose
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        if w_init_gain is not None:
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.transpose or x.size(1) != self.in_channels:
            assert x.size(2) == self.in_channels
            x = x.transpose(1, 2)
            self.transpose = True

        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        outputs = self.activation(self.bn(super(Conv1d, self).forward(x)))
        return outputs.transpose(1, 2) if self.transpose else outputs

    def extra_repr(self):
        return '(settings): {}\n(causal): {}\n(input_transpose): {}'.format(
                super(Conv1d, self).extra_repr(), self.causal, self.transpose)

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


class WN(torch.nn.Module):
  def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, causal=False):
    super(WN, self).__init__()
    assert(kernel_size % 2 == 1)
    self.hidden_channels =hidden_channels
    self.kernel_size = kernel_size,
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout

    self.in_layers = torch.nn.ModuleList()
    self.res_skip_layers = torch.nn.ModuleList()
    self.drop = nn.Dropout(p_dropout)

    if gin_channels != 0:
      cond_layer = Conv1d(gin_channels, 2*hidden_channels*n_layers, 1, causal=causal)
      self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

    for i in range(n_layers):
      dilation = dilation_rate ** i
      in_layer = Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                 dilation=dilation, causal=causal)
      in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
      self.in_layers.append(in_layer)

      # last one is not necessary
      if i < n_layers - 1:
        res_skip_channels = 2 * hidden_channels
      else:
        res_skip_channels = hidden_channels

      res_skip_layer = Conv1d(hidden_channels, res_skip_channels, 1, causal=causal)
      res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
      self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask, g=None, **kwargs):
    output = torch.zeros_like(x)
    n_channels_tensor = torch.IntTensor([self.hidden_channels])

    if g is not None:
      g = self.cond_layer(g)

    for i in range(self.n_layers):
      x_in = self.in_layers[i](x)
      if g is not None:
        cond_offset = i * 2 * self.hidden_channels
        g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
      else:
        g_l = torch.zeros_like(x_in)

      acts = fused_add_tanh_sigmoid_multiply(
          x_in,
          g_l,
          n_channels_tensor)
      acts = self.drop(acts)

      res_skip_acts = self.res_skip_layers[i](acts)
      if i < self.n_layers - 1:
        res_acts = res_skip_acts[:,:self.hidden_channels,:]
        x = (x + res_acts) * x_mask
        output = output + res_skip_acts[:,self.hidden_channels:,:]
      else:
        output = output + res_skip_acts
    return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)


class ResidualCouplingLayer(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
      causal=True):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = Conv1d(self.half_channels, hidden_channels, 1, causal=causal)
    self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels, causal=causal)
    self.post = Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1, causal=causal)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h = self.enc(h, x_mask, g=g)
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

class Flip(nn.Module):
  def forward(self, x, *args, reverse=False, **kwargs):
    x = torch.flip(x, [1])
    if not reverse:
      logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
      return x, logdet
    else:
      return x


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0,
      causal=True):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, causal=causal))
      self.flows.append(Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

