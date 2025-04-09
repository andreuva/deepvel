import math
import torch
from torch import nn

### Activation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

### Multi-Time Embedding Module
class MultiTimeEmbedding(nn.Module):
    """
    Computes a conditioning embedding from three time values by using the differences
    (dt1 = t_middle - t_first and dt2 = t_last - t_middle). These differences are then
    sinusoidally embedded, concatenated, and processed through an MLP.
    """
    def __init__(self, emb_dim):
        # emb_dim should be even.
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            Swish(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, t):
        # t: tensor of shape [B, 3] where t[:,0] = t_first, t[:,1] = t_middle, t[:,2] = t_last
        dt1 = t[:, 1] - t[:, 0]  # Difference between middle and first
        dt2 = t[:, 2] - t[:, 1]  # Difference between last and middle

        # For each difference we compute a sinusoidal embedding.
        # We want each dt to be embedded into a vector of size emb_dim/2.
        d = self.emb_dim // 2

        def sinusoidal_embedding(x, d):
            # x: shape [B]
            half = d // 2  # we use half for sin and half for cos components
            if half < 1:
                half = 1
            # Create a frequency vector.
            frequencies = torch.exp(torch.arange(half, dtype=torch.float32, device=x.device) *
                                      (-math.log(10000) / (half - 1)))
            # x[:, None]: shape [B, 1]. Multiply by frequencies to get [B, half]
            x_proj = x[:, None] * frequencies[None, :]
            sin_emb = torch.sin(x_proj)
            cos_emb = torch.cos(x_proj)
            return torch.cat([sin_emb, cos_emb], dim=-1)  # shape [B, d] if d == 2 * half

        emb_dt1 = sinusoidal_embedding(dt1, d)
        emb_dt2 = sinusoidal_embedding(dt2, d)

        # Concatenate both embeddings -> shape [B, emb_dim]
        t_emb = torch.cat([emb_dt1, emb_dt2], dim=-1)
        t_emb = self.mlp(t_emb)
        return t_emb

### Residual Block with Time Conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 n_groups: int = 32, dropout: float = 0.1, padding_mode: str = 'zeros'):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode=padding_mode)
        else:
            self.shortcut = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        # Project time embedding to channel space.
        self.time_dense = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time conditioning: expand t_emb and add to h.
        t_out = self.time_dense(t_emb)[:, :, None, None]
        h = h + t_out
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

### Multi-head Attention Block (unchanged)
class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(1, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        batch_size, n_channels, height, width = x.shape
        x_reshaped = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x_reshaped).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x_reshaped
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res

### DownBlock – now passing time embedding to ResidualBlock
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, time_emb_dim: int,
                 padding_mode: str = 'zeros'):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x

### UpBlock – concatenates skip connection and applies time conditioning.
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool, time_emb_dim: int,
                 padding_mode: str = 'zeros'):
        super().__init__()
        # Note: input channels become in_channels + out_channels when concatenating skip connection.
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = torch.cat((x, skip), dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x

### MiddleBlock – applies two ResidualBlocks with attention, each receives t_emb.
class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_emb_dim: int, padding_mode: str = 'zeros'):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_emb_dim, padding_mode=padding_mode)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x

### Upsample and Downsample (unchanged)
class Upsample(nn.Module):
    def __init__(self, n_channels, bilinear: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        else:
            self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        if self.bilinear:
            x = self.up(x)
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_channels, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
    
    def forward(self, x):
        return self.conv(x)

### The Adapted U-Net for Velocity Prediction with Multi-Time Embeddings
class UNetForVelocity(nn.Module):
    """
    U-Net that predicts horizontal velocities (v_x, v_y) from 3 input images.
    It uses a multi-time embedding module that computes the differences between times.
    """
    def __init__(self, input_channels: int = 3, output_channels: int = 2, n_channels: int = 64,
                 ch_mults: tuple = (1, 2, 2, 4),
                 is_attn: tuple = (False, False, True, True),
                 n_blocks: int = 2,
                 time_emb_dim: int = 64,
                 padding_mode: str = 'zeros',
                 bilinear: bool = False):
        super().__init__()
        # Multi-time embedding: expects time input of shape [B, 3]
        self.time_embed = MultiTimeEmbedding(time_emb_dim)
        
        # Project 3 images stacked along channels to initial features.
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        n_resolutions = len(ch_mults)
        self.down = nn.ModuleList()
        in_ch = n_channels
        for i in range(n_resolutions):
            out_ch = in_ch * ch_mults[i]
            for _ in range(n_blocks):
                self.down.append(DownBlock(in_ch, out_ch, is_attn[i], time_emb_dim, padding_mode=padding_mode))
                in_ch = out_ch
            if i < n_resolutions - 1:
                self.down.append(Downsample(in_ch, padding_mode=padding_mode))
        
        self.middle = MiddleBlock(in_ch, time_emb_dim, padding_mode=padding_mode)
        
        self.up = nn.ModuleList()
        for i in reversed(range(n_resolutions)):
            for _ in range(n_blocks):
                self.up.append(UpBlock(in_ch, in_ch, is_attn[i], time_emb_dim, padding_mode=padding_mode))
            out_ch = in_ch // ch_mults[i]
            self.up.append(UpBlock(in_ch, out_ch, is_attn[i], time_emb_dim, padding_mode=padding_mode))
            in_ch = out_ch
            if i > 0:
                self.up.append(Upsample(in_ch, bilinear=bilinear, padding_mode=padding_mode))
        
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
    
    def forward(self, x_in, t):
        """
        Args:
          x_in: Tensor of shape [B, 3, H, W] representing 3 consecutive images.
          t: Tensor of shape [B, 3] containing the three time values for the images.
        """
        # Compute time conditioning from the differences.
        t_emb = self.time_embed(t)  # shape [B, time_emb_dim]
        
        # Process the images.
        x = self.image_proj(x_in)
        skip_connections = [x]
        downs = []
        for module in self.down:
            if isinstance(module, DownBlock):
                x = module(x, t_emb)
                downs.append(x)
            else:
                x = module(x)
                downs.append(x)
        
        x = self.middle(x, t_emb)
        
        for module in self.up:
            if isinstance(module, Upsample):
                x = module(x)
            else:
                # Use a saved skip connection.
                skip = downs.pop()
                x = module(x, skip, t_emb)
        
        out = self.final(self.act(self.norm(x)))
        return out

### Testing the network.
if __name__ == '__main__':
    model = UNetForVelocity(input_channels=3, output_channels=2, n_channels=32,
                            ch_mults=(1, 2, 2, 2), is_attn=(False, False, False, False),
                            n_blocks=1, time_emb_dim=32)
    print(model)
    
    # Dummy input: batch size 2, 3 images of size 64x64.
    x = torch.zeros((2, 3, 64, 64))
    # Dummy times: one time stamp per image per batch (shape [B,3]).
    t = torch.tensor([[0.0, 0.5, 1.0],
                      [0.0, 0.5, 1.0]])
    out = model(x, t)
    print("Output shape:", out.shape)
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
