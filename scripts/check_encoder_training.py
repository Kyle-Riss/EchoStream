#!/usr/bin/env python3
"""
Check if encoder is actually being trained.
"""
import sys
sys.path.insert(0, '.')
import torch
from models.echostream_model import EchoStreamConfig, build_echostream_model

def check_encoder_training():
    """Check encoder training status."""
    
    # Load model
    config = EchoStreamConfig()
    config.encoder_embed_dim = 128
    config.encoder_layers = 4
    config.encoder_ffn_embed_dim = 512
    config.decoder_embed_dim = 128
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 2
    config.st_decoder_layers = 2
    config.dropout = 0.1
    
    model = build_echostream_model(config)
    
    print("="*70)
    print("🔍 Encoder Training Status Check")
    print("="*70)
    
    # 1. Check requires_grad
    print("\n1️⃣ Parameter requires_grad status:")
    print("-"*70)
    
    encoder_params = []
    st_decoder_params = []
    other_params = []
    
    for name, p in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append((name, p))
            status = "✅ TRAINABLE" if p.requires_grad else "❌ FROZEN"
            print(f"{status}: {name} {list(p.shape)}")
        elif 'st_ctc_decoder' in name or 'st_decoder' in name:
            st_decoder_params.append((name, p))
    
    print(f"\n📊 Encoder parameters: {len(encoder_params)}")
    print(f"📊 ST decoder parameters: {len(st_decoder_params)}")
    
    # Check if any encoder params are frozen
    frozen_encoder = [name for name, p in encoder_params if not p.requires_grad]
    if frozen_encoder:
        print(f"\n❌ WARNING: {len(frozen_encoder)} encoder params are FROZEN!")
        for name in frozen_encoder[:5]:
            print(f"  - {name}")
    else:
        print(f"\n✅ All encoder parameters are trainable")
    
    # 2. Check optimizer param groups
    print("\n2️⃣ Optimizer parameter groups:")
    print("-"*70)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    total_params = 0
    encoder_in_opt = 0
    st_decoder_in_opt = 0
    
    for i, g in enumerate(optimizer.param_groups):
        group_params = 0
        for p in g["params"]:
            group_params += p.numel()
            total_params += p.numel()
        print(f"Group {i}: lr={g['lr']}, num_params={group_params:,}")
    
    # Count encoder params in optimizer
    encoder_param_ids = {id(p) for _, p in encoder_params}
    st_decoder_param_ids = {id(p) for _, p in st_decoder_params}
    
    for g in optimizer.param_groups:
        for p in g["params"]:
            if id(p) in encoder_param_ids:
                encoder_in_opt += p.numel()
            if id(p) in st_decoder_param_ids:
                st_decoder_in_opt += p.numel()
    
    encoder_total = sum(p.numel() for _, p in encoder_params if p.requires_grad)
    st_decoder_total = sum(p.numel() for _, p in st_decoder_params if p.requires_grad)
    
    print(f"\n📊 Encoder params in optimizer: {encoder_in_opt:,} / {encoder_total:,}")
    print(f"📊 ST decoder params in optimizer: {st_decoder_in_opt:,} / {st_decoder_total:,}")
    
    if encoder_in_opt < encoder_total:
        print(f"\n❌ WARNING: Not all encoder params are in optimizer!")
        print(f"   Missing: {encoder_total - encoder_in_opt:,} params")
    else:
        print(f"\n✅ All encoder parameters are in optimizer")
    
    print("\n" + "="*70)
    print("🎯 Diagnosis:")
    
    if frozen_encoder:
        print(f"  ❌ Encoder has frozen parameters!")
        print(f"     → This is likely the root cause of blank collapse")
        print(f"     → Encoder is not learning, so ST decoder has no signal")
    elif encoder_in_opt < encoder_total:
        print(f"  ❌ Encoder params missing from optimizer!")
        print(f"     → This is likely the root cause of blank collapse")
    else:
        print(f"  ✅ Encoder is properly set up for training")
        print(f"     → Problem is elsewhere (encoder output, ST path, etc.)")
    
    print("="*70)


if __name__ == "__main__":
    check_encoder_training()


