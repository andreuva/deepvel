import math
import torch
from torch import nn
from torchsummary import summary

### Activation function (prebuilt Swish)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

### Multi-Time Embedding Module
class MultiTimeEmbedding(nn.Module):
    """
    Computes a conditioning embedding from three time values by computing the differences
    (dt₁ = t_middle - t_first and dt₂ = t_last - t_middle), embedding each with a sinusoidal embedding,
    concatenating them, and then processing them with an MLP.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            Swish(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        
    def forward(self, t):
        # t: tensor of shape [B, 3] where each sample is [t_first, t_middle, t_last]
        dt1 = t[:, 1] - t[:, 0]  # time difference between middle and first frame
        dt2 = t[:, 2] - t[:, 1]  # time difference between last and middle frame
        d = self.emb_dim // 2   # allocate half for each dt
        
        def sinusoidal_embedding(x, d):
            # x: [B]
            half = d // 2 if d // 2 >= 1 else 1
            frequencies = torch.exp(torch.arange(half, dtype=torch.float32, device=x.device) *
                                      (-math.log(10000) / (half - 1)))
            x_proj = x[:, None] * frequencies[None, :]  # shape: [B, half]
            sin_emb = torch.sin(x_proj)
            cos_emb = torch.cos(x_proj)
            return torch.cat([sin_emb, cos_emb], dim=-1)  # shape: [B, d] if d == 2 * half
        
        emb_dt1 = sinusoidal_embedding(dt1, d)
        emb_dt2 = sinusoidal_embedding(dt2, d)
        # Concatenate embeddings: shape becomes [B, emb_dim]
        t_emb = torch.cat([emb_dt1, emb_dt2], dim=-1)
        return self.mlp(t_emb)

### Prebaked Attention Block using nn.MultiheadAttention
class PrebakedAttentionBlock(nn.Module):
    """
    A wrapper around nn.MultiheadAttention that applies it to image features.
    The input is reshaped from [B, C, H, W] into sequence form, then attention is computed,
    and finally the output is reshaped back.
    """
    def __init__(self, n_channels, n_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.ln = nn.LayerNorm(n_channels)
        self.mha = nn.MultiheadAttention(embed_dim=n_channels, num_heads=n_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions: shape [seq_len, B, C]
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)
        # Apply layer normalization on the last dimension (channels)
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        # Add residual connection
        out_flat = x_flat + attn_out
        # Reshape back to [B, C, H, W]
        out = out_flat.permute(1, 2, 0).view(B, C, H, W)
        return out

### Residual Block with time-conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 n_groups: int = 32, dropout: float = 0.1, padding_mode: str = 'zeros'):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
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
        # Inject time conditioning: project time embedding and add (expand spatially).
        t_out = self.time_dense(t_emb)[:, :, None, None]
        h = h + t_out
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

### DownBlock – injects time embedding and uses prebaked attention if required.
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool,
                 time_emb_dim: int, padding_mode: str = 'zeros'):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = PrebakedAttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x

### UpBlock – concatenates skip connection, uses ResidualBlock with time conditioning,
### and applies attention if needed.
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, has_attn: bool,
                 time_emb_dim: int, padding_mode: str = 'zeros'):
        super().__init__()
        # Note: input channels become (in_channels + out_channels) after concatenating the skip connection.
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = PrebakedAttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = torch.cat((x, skip), dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x

### MiddleBlock – applies two ResidualBlocks with a prebaked attention block in between.
class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_emb_dim: int, padding_mode: str = 'zeros'):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = PrebakedAttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_emb_dim, padding_mode=padding_mode)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x

### Upsample and Downsample layers (using standard conv layers)
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

### The Adapted U‑Net for Velocity Prediction with Multi-Time Embeddings
class UNetForVelocity(nn.Module):
    """
    U‑Net that predicts horizontal velocities (vₓ, v_y) from 3 input images.
    It uses multi‑time embeddings (computed from the differences between the times of the frames)
    and injects these into every residual block. Prebaked attention blocks (wrapping nn.MultiheadAttention)
    are used at selected resolutions.
    """
    def __init__(self, input_channels: int = 3, output_channels: int = 2, n_channels: int = 64,
                 ch_mults: tuple = (1, 2, 2, 4),
                 is_attn: tuple = (False, False, True, True),
                 n_blocks: int = 2, time_emb_dim: int = 64,
                 padding_mode: str = 'zeros', bilinear: bool = False):
        super().__init__()
        # Multi-time embedding: expects time tensor of shape [B, 3]
        self.time_embed = MultiTimeEmbedding(time_emb_dim)
        
        # Project 3-channel (3 images) input to n_channels.
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
            t: Tensor of shape [B, 3] containing the time values of [t_first, t_mid, t_last].
        """
        # Compute time conditioning from time differences.
        t_emb = self.time_embed(t)  # shape: [B, time_emb_dim]
        
        # Process the image input and initialize skip connections.
        x = self.image_proj(x_in)
        downs = [x]  # initialize "downs" with the initial projected features
        
        # Encoder path: pass through all "down" modules.
        for module in self.down:
            if isinstance(module, DownBlock):
                x = module(x, t_emb)
                downs.append(x)
            else:
                x = module(x)
                downs.append(x)
                
        # Middle block processing.
        x = self.middle(x, t_emb)
        
        # Decoder path: use skip connections from downs.
        for module in self.up:
            if isinstance(module, Upsample):
                x = module(x)
            else:
                # Pop a skip connection. With the initialization, downs will have enough elements.
                skip = downs.pop()
                x = module(x, skip, t_emb)
                
        out = self.final(self.act(self.norm(x)))
        return out

### Testing the network.
if __name__ == '__main__':
    model = UNetForVelocity(input_channels=3, output_channels=2, n_channels=32,
                            ch_mults=(1, 2, 2, 2), is_attn=(False, False, False, False),
                            n_blocks=1, time_emb_dim=32)
    
    batch_size=2
    in_chanels=3
    in_size = (batch_size, in_chanels, 64, 64)
    
    # Dummy input: batch size 2, 3 images of size 64x64.
    x = torch.zeros(in_size)
    # Dummy time input: one time stamp per frame per sample (shape [B, 3]).
    t = torch.tensor([[0.0, 0.5, 1.0],
                      [0.0, 0.5, 1.0]])
    
    print(model)
    
    out = model(x, t)
    print("Output shape:", out.shape)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
