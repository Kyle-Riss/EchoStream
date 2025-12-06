"""
Conformer Encoder for EchoStream.
Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""
import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional


class ConformerConvolutionModule(nn.Module):
    """
    Conformer Convolution Module.
    
    Architecture:
        LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout
    """
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Pointwise convolution 1
        self.pointwise_conv1 = nn.Conv1d(
            embed_dim,
            embed_dim * 2,  # For GLU
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        # GLU activation
        self.glu = nn.GLU(dim=1)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=embed_dim,  # Depthwise
        )
        
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.SiLU()  # Swish
        
        # Pointwise convolution 2
        self.pointwise_conv2 = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        
        Returns:
            [B, T, D]
        """
        # Layer norm
        x = self.layer_norm(x)
        
        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        # Pointwise conv 1 + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        # Pointwise conv 2
        x = self.pointwise_conv2(x)
        
        # Transpose back: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed-Forward Module.
    
    Architecture:
        LayerNorm → Linear → Swish → Dropout → Linear → Dropout
    """
    
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.activation = nn.SiLU()  # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        
        Returns:
            [B, T, D]
        """
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConformerEncoderLayer(nn.Module):
    """
    Conformer Encoder Layer.
    
    Architecture:
        x → FFN1 (0.5 residual) → Self-Attention → Conv Module → FFN2 (0.5 residual) → LayerNorm → x
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feed-forward modules
        self.ffn1 = ConformerFeedForward(embed_dim, ffn_embed_dim, dropout)
        self.ffn2 = ConformerFeedForward(embed_dim, ffn_embed_dim, dropout)
        
        # Self-attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # Convolution module
        self.conv_module = ConformerConvolutionModule(
            embed_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            padding_mask: [B, T] where True = padding
        
        Returns:
            [B, T, D]
        """
        # FFN1 with 0.5 residual
        residual = x
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x_attn, _ = self.self_attn(
            x, x, x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.self_attn_dropout(x_attn)
        
        # Convolution module
        residual = x
        x = residual + self.conv_module(x)
        
        # FFN2 with 0.5 residual
        residual = x
        x = x + 0.5 * self.ffn2(x)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder for EchoStream.
    
    Compatible with EchoStream's encoder interface.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        ffn_embed_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Subsampling (4x downsampling)
        # Simple linear subsampling for now
        self.subsample = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate subsampled feature dimension
        # After 2x Conv2d with stride=2: T/4, F/4
        subsampled_dim = (input_dim // 4) * 256
        self.linear_proj = nn.Linear(subsampled_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_embed_dim=ffn_embed_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Dict[str, List]:
        """
        Forward pass.
        
        Args:
            src_tokens: [B, T, F] input features
            src_lengths: [B] sequence lengths
        
        Returns:
            Dict with:
                - 'encoder_out': List of [T', B, D]
                - 'encoder_padding_mask': List of [B, T']
                - 'encoder_embedding': Empty list
                - 'encoder_states': Empty list
                - 'src_tokens': Empty list
                - 'src_lengths': Empty list
        """
        B, T, F = src_tokens.shape
        
        # Subsampling: [B, T, F] -> [B, T/4, F/4, 256]
        x = src_tokens.unsqueeze(1)  # [B, 1, T, F]
        x = self.subsample(x)  # [B, 256, T/4, F/4]
        
        # Reshape: [B, 256, T/4, F/4] -> [B, T/4, 256*F/4]
        B, C, T_sub, F_sub = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T/4, 256, F/4]
        x = x.view(B, T_sub, C * F_sub)  # [B, T/4, 256*F/4]
        
        # Linear projection
        x = self.linear_proj(x)  # [B, T/4, D]
        
        # Calculate subsampled lengths
        output_lengths = (src_lengths.float() / 4).ceil().long()
        
        # Create padding mask
        max_len = x.size(1)
        encoder_padding_mask = (
            torch.arange(max_len, device=src_lengths.device).unsqueeze(0) 
            >= output_lengths.unsqueeze(1)
        )  # [B, T']
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Conformer layers
        for layer in self.layers:
            x = layer(x, padding_mask=encoder_padding_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Transpose to [T, B, D] (EchoStream/Fairseq format)
        encoder_out = x.transpose(0, 1)
        
        # Return in EchoStream format
        return {
            'encoder_out': [encoder_out],  # List of [T, B, D]
            'encoder_padding_mask': [encoder_padding_mask],  # List of [B, T]
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': [],
        }
    
    def reset_cache(self):
        """Reset cache (no-op for Conformer, but required for compatibility)."""
        pass


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Conformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        
        Returns:
            [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    print("Testing ConformerEncoder...")
    
    # Test
    encoder = ConformerEncoder(
        input_dim=80,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        ffn_embed_dim=512,
        conv_kernel_size=31,
        dropout=0.1,
    )
    
    # Dummy input
    B, T, F = 2, 100, 80
    src_tokens = torch.randn(B, T, F)
    src_lengths = torch.tensor([100, 80])
    
    # Forward
    output = encoder(src_tokens, src_lengths)
    
    print(f"Encoder output shape: {output['encoder_out'][0].shape}")  # [T', B, D]
    print(f"Padding mask shape: {output['encoder_padding_mask'][0].shape}")  # [B, T']
    print(f"Encoder output mean: {output['encoder_out'][0].mean().item():.4f}")
    print(f"Encoder output std: {output['encoder_out'][0].std().item():.4f}")
    
    # Check diversity
    enc = output['encoder_out'][0].transpose(0, 1)  # [B, T, D]
    x1 = enc[0].mean(dim=0)
    x2 = enc[1].mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(x1, x2, dim=0)
    print(f"Cosine similarity (sample 0 vs 1): {cos.item():.4f}")
    
    print("\n✅ ConformerEncoder test passed!")


