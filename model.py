import torch
import torch.nn as nn
import torch.nn.functional as F

class spatialProcess(nn.Module):
    def __init__(self, channels):
        super(spatialProcess, self).__init__()

        self.branch1 = nn.Conv2d(channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels//4, kernel_size=1),
            nn.Conv2d(channels//4, kernel_size=3, padding=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels//4, kernel_size=1),
            nn.Conv2d(channels//4, kernel_size=5, padding=2)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels//4, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        pool = self.pool(x)

        out = torch.cat([branch1, branch3, branch5, pool], dim=1)

        return out

class MultiHeadFrequencyAttentionWithChroma(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadFrequencyAttentionWithChroma, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Convolutional layer for chroma enhancement
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)  # RGB channels

    def rgb_to_ycbcr(self, rgb):

        matrix = torch.tensor(
            [[0.299, 0.587, 0.114],
             [-0.1687, -0.3313, 0.5],
             [0.5, -0.4187, -0.0813]],
            device=rgb.device
        ).T
        ycbcr = torch.matmul(rgb, matrix) + torch.tensor([0, 128, 128], device=rgb.device)
        return ycbcr

    def ycbcr_to_rgb(self, ycbcr):

        matrix = torch.tensor(
            [[1.0, 0.0, 1.402],
             [1.0, -0.344136, -0.714136],
             [1.0, 1.772, 0.0]],
            device=ycbcr.device
        )
        rgb = torch.matmul(ycbcr - torch.tensor([0, 128, 128], device=ycbcr.device), matrix.T)
        return rgb

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)

        # Step 1: FFT to frequency domain
        x_freq = fft.rfft(x, dim=-1)
        x_freq = torch.view_as_real(x_freq)

        # Step 2: Multi-Head Frequency Attention
        batch_size, seq_length, freq_dim, _ = x_freq.size()
        x_freq = x_freq.view(batch_size, seq_length, -1)
        attn_output, _ = self.attention(x_freq, x_freq, x_freq)
        output = self.linear(attn_output)
        output = self.norm(output + x_freq)

        # Step 3: Reshape and IFFT back to spatial domain
        output = output.view(batch_size, seq_length, freq_dim, 2)
        output = torch.view_as_complex(output)
        output = fft.irfft(output, dim=-1)

        # Step 4: Convert RGB to YCbCr
        ycbcr = self.rgb_to_ycbcr(output)

        # Step 5: Apply chroma enhancement using convolution
        ycbcr = ycbcr.permute(0, 2, 1).unsqueeze(1)  # Reshape for convolution
        enhanced_ycbcr = self.conv(ycbcr)
        enhanced_ycbcr = enhanced_ycbcr.squeeze(1).permute(0, 2, 1)

        # Step 6: Convert YCbCr back to RGB
        enhanced_output = self.ycbcr_to_rgb(enhanced_ycbcr)

        return enhanced_output

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4,
                 expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
                                       
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
