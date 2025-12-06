"""
EchoStream Speech Encoder

Integrates Emformer with speech preprocessing (Conv2D subsampling)
for efficient streaming speech-to-speech translation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emformer_layer import EmformerEncoder


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings (matching StreamSpeech Conformer).
    
    This is a simplified version that works with padding_mask input.
    """
    
    def __init__(
        self,
        max_positions: int = 6000,
        embedding_dim: int = 256,
        padding_idx: int = 1,
    ):
        super().__init__()
        
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Learnable positional embeddings
        self.weights = nn.Embedding(
            max_positions + padding_idx + 1, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize positional embeddings."""
        nn.init.normal_(self.weights.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weights.weight[self.padding_idx], 0)
    
    def forward(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            padding_mask: [B, T] where True = padding, False = valid
        
        Returns:
            positions: [T, B, D] positional embeddings (transposed for T-first format)
        """
        B, T = padding_mask.size()
        
        # Generate position indices: 0, 1, 2, ..., T-1 for each batch
        # Offset by padding_idx + 1
        positions = torch.arange(T, device=padding_mask.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, -1)  # [B, T]
        positions = positions + self.padding_idx + 1  # Offset
        
        # Clamp to valid range
        positions = positions.clamp(0, self.max_positions + self.padding_idx)
        
        # Get embeddings: [B, T, D]
        pos_emb = self.weights(positions)
        
        # Transpose to [T, B, D] format (T-first for StreamSpeech compatibility)
        pos_emb = pos_emb.transpose(0, 1)  # [T, B, D]
        
        return pos_emb


class Conv2dSubsampler(nn.Module):
    """
    Convolutional Subsampler for speech features.
    
    Uses two Conv2D layers with stride=2 to downsample by 4x.
    This is the standard preprocessing used in Conformer/Transformer speech models.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_feat_per_channel: int = 80,
        conv_out_channels: int = 256,
        encoder_embed_dim: int = 256,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.conv_out_channels = conv_out_channels
        self.encoder_embed_dim = encoder_embed_dim
        
        # Two Conv2D layers with stride=2 each → 4x downsampling
        self.conv1 = nn.Conv2d(
            input_channels,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        
        self.conv2 = nn.Conv2d(
            conv_out_channels,
            conv_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        
        # Calculate output feature dimension after convolution
        # Input: [B, C=1, T, F=80]
        # After conv1: [B, 256, T/2, F/2]
        # After conv2: [B, 256, T/4, F/4]
        output_feat_dim = (input_feat_per_channel // 4) * conv_out_channels
        
        # Linear projection to encoder dimension
        self.out_proj = nn.Linear(output_feat_dim, encoder_embed_dim)
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src_tokens: [B, T, F] or [B, C, T, F]
            src_lengths: [B]
        
        Returns:
            x: [T', B, D] where T' = T/4
            output_lengths: [B]
        """
        # Ensure 4D input [B, C, T, F]
        if src_tokens.dim() == 3:
            src_tokens = src_tokens.unsqueeze(1)  # [B, 1, T, F]
        
        B, C, T, F = src_tokens.size()
        
        # Conv layers
        x = self.conv1(src_tokens)  # [B, 256, T/2, F/2]
        x = nn.functional.relu(x)
        
        x = self.conv2(x)  # [B, 256, T/4, F/4]
        x = nn.functional.relu(x)
        
        # Reshape: [B, 256, T', F'] → [B, T', 256*F']
        B, C_out, T_out, F_out = x.size()
        x = x.permute(0, 2, 1, 3)  # [B, T', 256, F']
        x = x.reshape(B, T_out, C_out * F_out)  # [B, T', 256*F']
        
        # Project to encoder dimension
        x = self.out_proj(x)  # [B, T', D]
        
        # Convert to [T', B, D] format
        x = x.transpose(0, 1)  # [T', B, D]
        
        # Update lengths (downsampled by 4)
        output_lengths = ((src_lengths - 1) // 4 + 1).long()
        
        return x, output_lengths


class EchoStreamSpeechEncoder(nn.Module):
    """
    EchoStream Speech Encoder.
    
    Architecture:
        Speech Input [B, T, 80]
            ↓
        Conv2D Subsampling (4x downsample)
            ↓
        [T/4, B, 256]
            ↓
        Emformer Encoder (16 layers)
            ↓
        [T/4, B, 256]
    
    Key features:
    - Efficient streaming with Left Context Cache
    - Memory Bank for long-range dependencies
    - O(1) complexity per segment (vs O(T²) in Conformer)
    """
    
    def __init__(
        self,
        # Input parameters
        input_feat_per_channel: int = 80,
        input_channels: int = 1,
        
        # Encoder parameters
        encoder_embed_dim: int = 256,
        encoder_layers: int = 16,
        encoder_attention_heads: int = 4,
        encoder_ffn_embed_dim: int = 1024,
        
        # Emformer-specific parameters
        segment_length: int = 4,
        left_context_length: int = 30,
        right_context_length: int = 0,
        memory_size: int = 8,
        
        # Positional encoding
        max_source_positions: int = 6000,
        pos_enc_type: str = "abs",  # "abs" for absolute, "rel_pos" for relative
        
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        no_scale_embedding: bool = False,
    ):
        super().__init__()
        
        self.encoder_embed_dim = encoder_embed_dim
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.padding_idx = 1
        
        # Embed scale (matching StreamSpeech Conformer)
        self.embed_scale = math.sqrt(encoder_embed_dim) if not no_scale_embedding else 1.0
        
        # Subsampling layer
        self.subsample = Conv2dSubsampler(
            input_channels=input_channels,
            input_feat_per_channel=input_feat_per_channel,
            conv_out_channels=256,
            encoder_embed_dim=encoder_embed_dim,
        )
        
        # Positional embedding (matching StreamSpeech Conformer)
        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "abs":
            self.embed_positions = PositionalEmbedding(
                max_positions=max_source_positions,
                embedding_dim=encoder_embed_dim,
                padding_idx=self.padding_idx,
            )
        else:
            # For now, only support absolute positional encoding
            # Relative positional encoding can be added later if needed
            self.embed_positions = None
        
        # Linear projection (matching StreamSpeech Conformer)
        self.linear = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        
        # Dropout
        self.dropout_module = nn.Dropout(dropout)
        self.dropout = dropout
        
        # Emformer encoder
        self.emformer = EmformerEncoder(
            num_layers=encoder_layers,
            embed_dim=encoder_embed_dim,
            num_heads=encoder_attention_heads,
            segment_length=segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            memory_size=memory_size,
            ffn_embed_dim=encoder_ffn_embed_dim,
            dropout=dropout,
        )
    
    def reset_cache(self):
        """Reset Emformer cache for new utterance."""
        self.emformer.reset_cache()
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Dict[str, list]:
        """
        Forward pass.
        
        Args:
            src_tokens: Input features [B, T, 80]
            src_lengths: Sequence lengths [B]
        
        Returns:
            Dict with:
                - 'encoder_out': List of [T', B, D]
                - 'encoder_padding_mask': List of [B, T']
                - 'encoder_embedding': Empty list
                - 'encoder_states': Empty list
                - 'src_tokens': Empty list
                - 'src_lengths': Empty list
        """
        # Subsampling: [B, T, 80] → [T/4, B, 256]
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        
        # Create padding mask: [B, T'] where True = padding, False = valid
        # StreamSpeech format: True for padding positions
        max_len = x.size(0)  # T'
        encoder_padding_mask = (
            torch.arange(max_len, device=x.device).unsqueeze(0) >= input_lengths.unsqueeze(1)
        )  # [B, T']
        
        # Apply embed scale (matching StreamSpeech Conformer)
        x = self.embed_scale * x
        
        # Apply positional encoding (matching StreamSpeech Conformer)
        if self.embed_positions is not None:
            positions = self.embed_positions(encoder_padding_mask)  # [T', B, D]
            x = x + positions
        
        # Linear projection (matching StreamSpeech Conformer)
        x = self.linear(x)
        
        # Dropout (matching StreamSpeech Conformer)
        x = self.dropout_module(x)
        
        # Emformer encoding
        emformer_out = self.emformer(x, input_lengths)
        
        # Return in StreamSpeech/Fairseq format (required for decoder compatibility)
        # All decoders expect this exact format - DO NOT CHANGE!
        return {
            'encoder_out': emformer_out['encoder_out'],  # List of [T', B, D] - List format required!
            'encoder_padding_mask': emformer_out['encoder_padding_mask'],  # List of [B, T'] - List format required!
            'encoder_embedding': [],  # Empty list (required by StreamSpeech)
            'encoder_states': [],  # Empty list (required by StreamSpeech)
            'src_tokens': [],  # Empty list (required by StreamSpeech)
            'src_lengths': [],  # Empty list (required by StreamSpeech)
        }
    
    def reorder_encoder_out(self, encoder_out: Dict[str, list], new_order):
        """
        Reorder encoder output for beam search.
        
        Args:
            encoder_out: Output from forward()
            new_order: New order indices
        
        Returns:
            Reordered encoder_out
        """
        if len(encoder_out['encoder_out']) == 0:
            return encoder_out
        
        new_encoder_out = [
            encoder_out['encoder_out'][0].index_select(1, new_order)
        ]
        
        new_encoder_padding_mask = []
        if len(encoder_out['encoder_padding_mask']) > 0:
            new_encoder_padding_mask = [
                encoder_out['encoder_padding_mask'][0].index_select(0, new_order)
            ]
        
        return {
            'encoder_out': new_encoder_out,
            'encoder_padding_mask': new_encoder_padding_mask,
            'encoder_embedding': [],
            'encoder_states': [],
            'src_tokens': [],
            'src_lengths': [],
        }


if __name__ == "__main__":
    print("Testing EchoStream Speech Encoder...")
    
    # Create encoder
    encoder = EchoStreamSpeechEncoder(
        input_feat_per_channel=80,
        encoder_embed_dim=256,
        encoder_layers=4,  # Use 4 layers for faster testing
        encoder_attention_heads=4,
        segment_length=4,
        left_context_length=30,
    )
    
    # Test input
    B, T, F = 2, 100, 80
    src_tokens = torch.randn(B, T, F)
    src_lengths = torch.tensor([100, 80])
    
    # Forward
    encoder_out = encoder(src_tokens, src_lengths)
    
    print(f"Input shape: {src_tokens.shape}")
    print(f"Encoder output shape: {encoder_out['encoder_out'][0].shape}")
    print(f"Padding mask shape: {encoder_out['encoder_padding_mask'][0].shape}")
    print(f"Downsampling ratio: {T} → {encoder_out['encoder_out'][0].size(0)} (4x)")
    
    # Test cache reset
    encoder.reset_cache()
    print("\n✅ Cache reset successful")
    
    # Test streaming (multiple forward passes)
    print("\nTesting streaming mode...")
    encoder.reset_cache()
    
    chunk_size = 40
    for i in range(0, T, chunk_size):
        chunk_end = min(i + chunk_size, T)
        chunk_tokens = src_tokens[:, i:chunk_end, :]
        chunk_lengths = torch.tensor([chunk_end - i, min(chunk_end - i, 80 - i)])
        
        chunk_out = encoder(chunk_tokens, chunk_lengths)
        print(f"Chunk {i//chunk_size + 1}: Input {chunk_tokens.shape[1]} → Output {chunk_out['encoder_out'][0].size(0)}")
    
    print("\n✅ EchoStream Speech Encoder test passed!")

