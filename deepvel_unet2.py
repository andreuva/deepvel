import math
import torch
from torch import nn

class MultiTimeSinusoidalEmbedding(nn.Module):
    """
    Computes a conditioning embedding from multiple time values.
    
    Given an input tensor 't' of shape [B, T] (where T is the number of timestamps),
    this module:
      1. Computes differences between each adjacent timestamp.
      2. Applies a sinusoidal embedding to each difference.
      3. Concatenates all of the resulting embeddings.
      4. Processes the concatenated vector with an MLP to produce a final embedding of
         dimension `emb_dim`.
    
    Note:
      - For T timestamps there will be T-1 differences.
      - `emb_dim` should be divisible by (T-1) so that each difference gets the same number of dimensions.
    """
    def __init__(self, emb_dim, n_timestamps, dense_up_project=2):
        super().__init__()
        self.n_diffs = n_timestamps - 1  # number of adjacent differences
        if emb_dim % self.n_diffs != 0:
            raise ValueError("emb_dim must be divisible by (n_timestamps - 1)")
        self.diff_emb_dim = emb_dim // self.n_diffs
        
        # Create a simple MLP that processes the concatenated embeddings.
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * dense_up_project),
            nn.SiLU(),
            nn.Linear(emb_dim * dense_up_project, emb_dim)
        )
       
    def sinusoidal_embedding(self, x, d):
        """
        Computes a sinusoidal embedding for a tensor of time differences.
        
        Args:
            x: A tensor of shape [B] (a single time difference per batch element).
            d: An integer specifying the final embedding dimension for this difference.
               This value should be even (it is typically 2 * (d//2)).
        
        Returns:
            A tensor of shape [B, d] where the first half of the final dimension uses the sine
            function and the second half uses the cosine.
        """
        # Use half the dimensions for the sine and half for the cosine.
        half_dim = d // 2
        # Protect against a case where half_dim could be zero.
        if half_dim < 1:
            half_dim = 1
        # Compute the frequencies scaling factors.
        # Avoid division-by-zero by ensuring that if half_dim==1 the denominator is safely handled.
        scale = -math.log(10000) / (half_dim - 1) if half_dim > 1 else -math.log(10000)
        frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=x.device) * scale)
        
        # Multiply the time difference by each frequency.
        x_proj = x.unsqueeze(-1) * frequencies.unsqueeze(0)  # shape: [B, half_dim]
        sin_emb = torch.sin(x_proj)
        cos_emb = torch.cos(x_proj)
        # Concatenate to get a [B, d] embedding.
        return torch.cat([sin_emb, cos_emb], dim=-1)
    
    def forward(self, t):
        """
        Args:
            t: Tensor of shape [B, T] containing timestamp values.
        
        Returns:
            A tensor of shape [B, emb_dim] representing the final time embedding.
        """
        # Compute differences between consecutive timestamps: shape [B, T-1]
        dt = t[:, 1:] - t[:, :-1]
        embeddings = []
        
        # Compute a sinusoidal embedding for each time difference.
        for i in range(self.n_diffs):
            emb_i = self.sinusoidal_embedding(dt[:, i], self.diff_emb_dim)
            embeddings.append(emb_i)
        
        # Concatenate embeddings to form a tensor of shape [B, emb_dim]
        t_emb = torch.cat(embeddings, dim=-1)
        # Process with an MLP to capture interactions among differences.
        return self.mlp(t_emb)


### Attention Block using nn.MultiheadAttention
class SelfAttentionBlock(nn.Module):
    """
    A wrapper around nn.MultiheadAttention that applies it to image features.
    The input is reshaped from [B, C, H, W] into sequence form, then attention is computed,
    and finally the output is reshaped back.
    """
    def __init__(self, n_channels, n_heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(n_channels)
        self.mha = nn.MultiheadAttention(embed_dim=n_channels, num_heads=n_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten spatial dimensions: shape [seq_len, B, C]
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)
        # Apply layer normalization on the last dimension (channels)
        x_norm = self.ln(x_flat)
        # Apply self-attention (same tensor for query, key, value)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        # Add residual connection
        out_flat = x_flat + attn_out
        # Reshape back to [B, C, H, W]
        out = out_flat.permute(1, 2, 0).view(B, C, H, W)
        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=4):
        super(CrossAttentionBlock, self).__init__()
        self.ln = nn.LayerNorm(n_channels)
        self.attn = nn.MultiheadAttention(embed_dim=n_channels, num_heads=n_heads)
    
    def forward(self, query_feat, context_feat):
        # query_feat: [B, C, H, W] from frame 1
        # context_feat: [B, C, H, W] from frame 2
        B, C, H, W = query_feat.shape
        # Flatten and permute to [sequence_length, B, C]
        query_flat = query_feat.view(B, C, H * W).permute(2, 0, 1)
        context_flat = context_feat.view(B, C, H * W).permute(2, 0, 1)
        # Cross-attention: queries from frame1, keys and values from frame2
        attn_out, _ = self.attn(query_flat, context_flat, context_flat)
        # Reshape back to [B, C, H, W]
        attn_out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return attn_out


### Residual Block with time-conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 n_groups: int = 2, dropout: float = 0.0, padding_mode: str = 'zeros'):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
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
        self.attn = SelfAttentionBlock(out_channels) if has_attn else nn.Identity()

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
        self.attn = CrossAttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = torch.cat((x, skip), dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x, skip)
        return x


### MiddleBlock – applies two ResidualBlocks with a prebaked attention block in between.
class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_emb_dim: int, padding_mode: str = 'zeros'):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_emb_dim, padding_mode=padding_mode)
        self.attn = SelfAttentionBlock(n_channels)
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

###############################################################################
# Higher-Level Modules for Encoder and Decoder Levels
###############################################################################

class EncoderLevel(nn.Module):
    """
    Processes one resolution level in the encoder with a sequence of DownBlocks.
    """
    def __init__(self, in_channels, out_channels, n_blocks, has_attn, time_emb_dim, padding_mode='zeros'):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            current_in = in_channels if i == 0 else out_channels
            blocks.append(DownBlock(current_in, out_channels, has_attn, time_emb_dim, padding_mode=padding_mode))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        return x

class DecoderLevel(nn.Module):
    """
    Processes one resolution level in the decoder. The first block performs the skip
    concatenation (after projecting the skip to the expected number of channels if needed)
    and subsequent blocks refine the features.
    """
    def __init__(self, x_in_channels, skip_channels, out_channels, n_blocks, has_attn,
                 time_emb_dim, padding_mode='zeros'):
        super().__init__()
        # Project skip connection to the expected number of channels if needed.
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1,
                                   padding=0, padding_mode=padding_mode) if skip_channels != out_channels else nn.Identity()
        layers = []
        # First block uses skip concatenation.
        layers.append(UpBlock(x_in_channels, out_channels, has_attn, time_emb_dim, padding_mode=padding_mode))
        # Additional blocks (if any) refine the result without skip concatenation.
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, time_emb_dim, padding_mode=padding_mode))
        self.blocks = nn.ModuleList(layers)
    
    def forward(self, x, skip, t_emb):
        skip = self.skip_proj(skip)
        x = self.blocks[0](x, skip, t_emb)
        for block in self.blocks[1:]:
            x = block(x, t_emb)
        return x

###############################################################################
# The Reorganized U‑Net for Velocity Prediction (UDeepVel)
###############################################################################

class UDeepVel(nn.Module):
    """
    U‑Net that predicts horizontal velocities (v_x, v_y) from input images.
    This version uses:
      - A multi‑time embedding computed by differences with sinusoidal embeddings.
      - An encoder that saves one skip connection per resolution level.
      - A decoder that uses the saved skip connections (projected if needed) and upsamples.
      - Attention blocks are applied optionally.
    """
    def __init__(self,
                 input_channels: int = 3, 
                 output_channels: int = 2,
                 n_latent_chanels: int = 32,
                 chanels_multiples: tuple = (1, 2, 2, 4),
                 is_attn_encoder: tuple = (False, False, False, False),
                 is_attn_decoder: tuple = (True, True, True, True),
                 n_blocks: int = 2,
                 time_emb_dim: int = 32,
                 padding_mode: str = 'zeros',
                 bilinear: bool = False):

        super().__init__()
        # Multi-time embedding: expects time tensor of shape [B, input_channels]
        self.time_embed = MultiTimeSinusoidalEmbedding(time_emb_dim, input_channels)
        
        # Project input_channels to n_latent_chanels.
        self.image_proj = nn.Conv2d(input_channels, n_latent_chanels, kernel_size=3, padding=1, padding_mode=padding_mode)
        
        # Build encoder: each level outputs a skip connection.
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.encoder_channels = []  # Save the output channels at each level
        in_ch = n_latent_chanels
        for i, mult in enumerate(chanels_multiples):
            out_ch = in_ch * mult
            self.encoder_levels.append(
                EncoderLevel(in_ch, out_ch, n_blocks, is_attn_encoder[i], time_emb_dim, padding_mode=padding_mode)
            )
            self.encoder_channels.append(out_ch)
            in_ch = out_ch
            if i < len(chanels_multiples) - 1:
                self.downsamples.append(Downsample(in_ch, padding_mode=padding_mode))
        
        # Middle block.
        self.middle = MiddleBlock(in_ch, time_emb_dim, padding_mode=padding_mode)
        
        # Build decoder: process in reverse order.
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(chanels_multiples))):
            # For the decoder level, the skip connection comes from the encoder level i.
            skip_channels = self.encoder_channels[i]
            # Determine new number of channels.
            # Here we reverse the channel increase by dividing by the corresponding multiple.
            out_ch = in_ch // chanels_multiples[i]
            has_attn = is_attn_decoder[i]
            self.decoder_levels.append(
                DecoderLevel(in_ch, skip_channels, out_ch, n_blocks, has_attn, time_emb_dim, padding_mode=padding_mode)
            )
            in_ch = out_ch
            if i > 0:
                self.upsamples.append(Upsample(in_ch, bilinear=bilinear, padding_mode=padding_mode))
        
        # Final normalization and projection.
        self.norm = nn.GroupNorm(8, in_ch)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x_in, t):
        """
        Args:
            x_in: Tensor of shape [B, 3, H, W] representing the input images.
            t: Tensor of shape [B, input_channels] with time stamps.
        """
        t_emb = self.time_embed(t)  # Time conditioning embedding.
        x = self.image_proj(x_in)
        
        # Encoder pass: process each resolution level and save skip connections.
        skip_connections = []
        for i, encoder in enumerate(self.encoder_levels):
            x = encoder(x, t_emb)
            skip_connections.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        
        # Middle block.
        x = self.middle(x, t_emb)
        
        # Decoder pass: use saved skip connections in reverse order.
        # The skip_connections list order matches the encoder order; we pop from the end.
        for i, decoder in enumerate(self.decoder_levels):
            skip = skip_connections.pop()
            x = decoder(x, skip, t_emb)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)
        
        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x

###############################################################################
# Testing the Model
###############################################################################

if __name__ == '__main__':
    batch_size = 2
    in_channels = 2   # Three-channel input image (or consecutive frames, etc.)
    h_size, w_size = 64, 64
    time_dim = 16     # Expecting a time tensor of shape [B, in_channels]
    x_in = torch.zeros(batch_size, in_channels, h_size, w_size)
    t = torch.zeros(batch_size, in_channels)  # Dummy time input
    
    model = UDeepVel(
        input_channels=in_channels, 
        output_channels=2, 
        n_latent_chanels=16,
        chanels_multiples=(1, 2, 2, 4),
        is_attn_encoder=(False, False, False , False),
        is_attn_decoder=(True, True, True, True),
        n_blocks=2,
        time_emb_dim=time_dim,
        padding_mode='zeros',
        bilinear=False
    )

    out = model(x_in, t)

    from torchinfo import summary
    summary(model,
            input_data=(x_in, t),
            col_names=(                
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "trainable"
                ),
            col_width=20,
            depth=8
            )
