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


class BigVGANDecoder(torch.nn.Module):
    def __init__(self, h):
        super(BigVGANDecoder, self).__init__()
        self.h = h
        causal = h.causal

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(h.latent_dim, h.upsample_initial_channel, 7, 1, causal=False))

        resblock = AMPBlock1 

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, causal=causal))
            ]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation, causal=causal))

        activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
        self.activation_post = Activation1d(activation=activation_post)
       
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)


    def inference_from_latents(self, x, do_sample=True):
        if self.h.use_vae and do_sample:
            assert x.size(1) == self.h.latent_dim * 2, "Input must be like [B, D, H]"
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
            m_p = torch.zeros_like(m_q)
            logs_p = torch.ones_like(logs_q)
        else:
            assert x.size(1) == self.h.latent_dim, "Input must be like [B, D, H]"

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
 

    def remove_weight_norm(self):
        print('removing weight norm ...')
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


