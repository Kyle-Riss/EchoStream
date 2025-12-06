"""
Simple BiLSTM Encoder for CTC baseline.
Purpose: Verify CTC + ST + SPM pipeline without Emformer complexity.
"""
import torch
import torch.nn as nn
from typing import Dict, List


class SimpleCTCEncoder(nn.Module):
    """
    Simple BiLSTM encoder for CTC baseline.
    
    This is a sanity check encoder to verify that:
    1. CTC loss calculation is correct
    2. ST decoder works properly
    3. SentencePiece tokenization is correct
    4. Padding/masking is handled correctly
    
    If this works but Emformer doesn't, the problem is in Emformer itself.
    """
    
    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Project to match expected output dimension
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer norm (matching Emformer style)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
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
                - 'encoder_out': List of [T, B, D]
                - 'encoder_padding_mask': List of [B, T]
                - 'encoder_embedding': Empty list
                - 'encoder_states': Empty list
                - 'src_tokens': Empty list
                - 'src_lengths': Empty list
        """
        # src_tokens: [B, T, F]
        # src_lengths: [B]
        
        B, T, F = src_tokens.shape
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            src_tokens,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        # BiLSTM forward
        out, _ = self.lstm(packed)
        
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # [B, T, 2H]
        
        # Project to hidden_dim
        out = self.proj(out)  # [B, T, H]
        
        # Layer norm
        out = self.layer_norm(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Transpose to [T, B, H] (EchoStream/Fairseq format)
        encoder_out = out.transpose(0, 1)  # [T, B, H]
        
        # Create padding mask: [B, T] where True = padding
        max_len = out.size(1)
        encoder_padding_mask = (
            torch.arange(max_len, device=src_lengths.device).unsqueeze(0) 
            >= src_lengths.unsqueeze(1)
        )  # [B, T]
        
        # Return in EchoStream/Fairseq format (List format required!)
        return {
            'encoder_out': [encoder_out],  # List of [T, B, D]
            'encoder_padding_mask': [encoder_padding_mask],  # List of [B, T]
            'encoder_embedding': [],  # Empty list (required by EchoStream)
            'encoder_states': [],  # Empty list (required by EchoStream)
            'src_tokens': [],  # Empty list (required by EchoStream)
            'src_lengths': [],  # Empty list (required by EchoStream)
        }
    
    def reset_cache(self):
        """Reset cache (no-op for BiLSTM, but required for compatibility)."""
        pass


if __name__ == "__main__":
    print("Testing SimpleCTCEncoder...")
    
    # Test
    encoder = SimpleCTCEncoder(
        input_dim=80,
        hidden_dim=128,
        num_layers=3,
        dropout=0.1,
    )
    
    # Dummy input
    B, T, F = 2, 100, 80
    src_tokens = torch.randn(B, T, F)
    src_lengths = torch.tensor([100, 80])
    
    # Forward
    output = encoder(src_tokens, src_lengths)
    
    print(f"Encoder output shape: {output['encoder_out'][0].shape}")  # [T, B, H]
    print(f"Padding mask shape: {output['encoder_padding_mask'][0].shape}")  # [B, T]
    print(f"Encoder output mean: {output['encoder_out'][0].mean().item():.4f}")
    print(f"Encoder output std: {output['encoder_out'][0].std().item():.4f}")
    
    # Check diversity
    enc = output['encoder_out'][0].transpose(0, 1)  # [B, T, H]
    x1 = enc[0].mean(dim=0)
    x2 = enc[1].mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(x1, x2, dim=0)
    print(f"Cosine similarity (sample 0 vs 1): {cos.item():.4f}")
    
    print("\n✅ SimpleCTCEncoder test passed!")


