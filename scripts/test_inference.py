#!/usr/bin/env python3
"""
Simple inference test script for EchoStream.
Tests model with random audio data.
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))

from echostream_model import build_echostream_model, EchoStreamConfig
from datasets import SpeechFeatureExtractor
from datasets.s2st_dataset import _load_global_cmvn

def test_inference(
    config_path: str = "configs/echostream_config.mini.yaml",
    checkpoint_path: str = "checkpoints_mini_units_v4/checkpoint_best.pt",
    use_random_audio: bool = True,
    audio_path: Optional[str] = None,
    audio_duration_sec: float = 2.0,
):
    """Test inference with random or real audio."""
    
    print("="*70)
    print("EchoStream Inference Test")
    print("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Build config
    config_overrides = {}
    encoder_cfg = config_dict.get('encoder', {})
    if encoder_cfg:
        if 'embed_dim' in encoder_cfg:
            config_overrides['encoder_embed_dim'] = encoder_cfg['embed_dim']
        if 'layers' in encoder_cfg:
            config_overrides['encoder_layers'] = encoder_cfg['layers']
        if 'attention_heads' in encoder_cfg:
            config_overrides['encoder_attention_heads'] = encoder_cfg['attention_heads']
        if 'ffn_embed_dim' in encoder_cfg:
            config_overrides['encoder_ffn_embed_dim'] = encoder_cfg['ffn_embed_dim']
        if 'segment_length' in encoder_cfg:
            config_overrides['segment_length'] = encoder_cfg['segment_length']
        if 'left_context_length' in encoder_cfg:
            config_overrides['left_context_length'] = encoder_cfg['left_context_length']
        if 'right_context_length' in encoder_cfg:
            config_overrides['right_context_length'] = encoder_cfg['right_context_length']
        if 'memory_size' in encoder_cfg:
            config_overrides['memory_size'] = encoder_cfg['memory_size']
        if 'input_feat_per_channel' in encoder_cfg:
            config_overrides['input_feat_per_channel'] = encoder_cfg['input_feat_per_channel']
    
    mt_decoder_cfg = config_dict.get('mt_decoder', {})
    if mt_decoder_cfg:
        if 'embed_dim' in mt_decoder_cfg:
            config_overrides['decoder_embed_dim'] = mt_decoder_cfg['embed_dim']
        if 'layers' in mt_decoder_cfg:
            config_overrides['mt_decoder_layers'] = mt_decoder_cfg['layers']
    
    unit_decoder_cfg = config_dict.get('unit_decoder', {})
    if unit_decoder_cfg:
        if 'embed_dim' in unit_decoder_cfg and 'decoder_embed_dim' not in config_overrides:
            config_overrides['decoder_embed_dim'] = unit_decoder_cfg['embed_dim']
        if 'layers' in unit_decoder_cfg:
            config_overrides['unit_decoder_layers'] = unit_decoder_cfg['layers']
        if 'num_units' in unit_decoder_cfg:
            config_overrides['num_units'] = unit_decoder_cfg['num_units']
    
    st_decoder_cfg = config_dict.get('st_decoder', {})
    if st_decoder_cfg and 'layers' in st_decoder_cfg:
        config_overrides['st_decoder_layers'] = st_decoder_cfg['layers']
    
    training_cfg = config_dict.get('training', {})
    if training_cfg:
        if 'dropout' in training_cfg:
            config_overrides['dropout'] = training_cfg['dropout']
        if 'attention_dropout' in training_cfg:
            config_overrides['attention_dropout'] = training_cfg['attention_dropout']
        if 'activation_dropout' in training_cfg:
            config_overrides['activation_dropout'] = training_cfg['activation_dropout']
    
    config = EchoStreamConfig.from_dict(config_overrides)
    print(f"Model: {config.encoder_layers}L Emformer + Decoders")
    print(f"  Encoder dim: {config.encoder_embed_dim}")
    print(f"  Decoder dim: {config.decoder_embed_dim}")
    
    # Build model
    print("\nBuilding model...")
    model = build_echostream_model(config)
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Using random initialization")
    else:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out vocoder keys if needed
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith("vocoder.")}
        
        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)} (vocoder excluded)")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print("✅ Checkpoint loaded successfully")
    
    # Prepare input
    data_cfg = config_dict.get('data', {})
    sample_rate = data_cfg.get('sample_rate', 16000)
    num_mel_bins = data_cfg.get('num_mel_bins', 80)
    
    print(f"\nPreparing input audio...")
    if audio_path and Path(audio_path).exists():
        # Load real audio file
        import soundfile as sf
        import librosa
        waveform, orig_sr = sf.read(audio_path)
        waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        else:
            waveform = waveform[:1]  # Take first channel if stereo
        
        # Resample if needed
        if orig_sr != sample_rate:
            waveform_np = waveform.squeeze(0).numpy()
            waveform_np = librosa.resample(waveform_np, orig_sr=orig_sr, target_sr=sample_rate)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        print(f"   Loaded audio: {audio_path}")
        print(f"   Duration: {waveform.size(1) / sample_rate:.2f}s ({waveform.size(1)} samples)")
    elif use_random_audio:
        # Generate random audio waveform
        num_samples = int(audio_duration_sec * sample_rate)
        waveform = torch.randn(1, num_samples) * 0.1  # [1, T] normalized
        print(f"   Random audio: {num_samples} samples ({audio_duration_sec:.1f}s)")
    else:
        raise ValueError("Either use_random_audio=True or provide audio_path")
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = SpeechFeatureExtractor(
        sample_rate=sample_rate,
        num_mel_bins=num_mel_bins,
        use_kaldi_fbank=True,
    )
    print(f"   Using Kaldi fbank: {feature_extractor.has_kaldi}")
    
    features = feature_extractor(waveform.squeeze(0), sample_rate)
    print(f"   Features shape: {features.shape} [time, mel]")
    
    # Apply CMVN if available
    cmvn_path = data_cfg.get('global_cmvn_stats_npz')
    if cmvn_path:
        cmvn_path = Path(cmvn_path)
        if cmvn_path.exists():
            print("\nApplying Global CMVN...")
            cmvn = _load_global_cmvn(cmvn_path)
            cmvn_mean, cmvn_std = cmvn
            eps = 1e-5
            features = (features - cmvn_mean) / (cmvn_std + eps)
            print("✅ CMVN applied")
        else:
            print(f"⚠️  CMVN file not found: {cmvn_path}")
    
    # Prepare model input
    src_tokens = features.unsqueeze(0).to(device)  # [1, T, 80]
    src_lengths = torch.tensor([features.size(0)], device=device)
    
    print(f"\nModel input:")
    print(f"   src_tokens: {src_tokens.shape}")
    print(f"   src_lengths: {src_lengths}")
    
    # Enable debug logging for MT decoder
    model._debug_logging = True
    
    # Run inference
    print("\n" + "="*70)
    print("Running inference...")
    print("="*70)
    
    with torch.no_grad():
        try:
            # Reset MT cache
            if hasattr(model, '_mt_prev_tokens'):
                model._mt_prev_tokens = None
            
            output = model(src_tokens=src_tokens, src_lengths=src_lengths)
            
            print("\n✅ Inference successful!")
            print(f"\nOutput keys: {list(output.keys())}")
            
            # Debug: Check ST CTC output
            if 'st_log_probs' in output and output['st_log_probs'] is not None:
                st_log_probs = output['st_log_probs']
                st_tokens_greedy = st_log_probs.argmax(dim=-1)  # [T, B]
                print(f"\n🔍 ST CTC Debug:")
                print(f"   ST log_probs shape: {st_log_probs.shape}")
                print(f"   ST tokens greedy shape: {st_tokens_greedy.shape}")
                print(f"   First 20 tokens: {st_tokens_greedy[:20, 0].tolist()}")
                
                # Count blanks and pads
                blank_count = (st_tokens_greedy == 0).sum().item()
                pad_count = (st_tokens_greedy == 1).sum().item()
                total = st_tokens_greedy.numel()
                print(f"   Blank (0) count: {blank_count}/{total} ({100*blank_count/total:.1f}%)")
                print(f"   Pad (1) count: {pad_count}/{total} ({100*pad_count/total:.1f}%)")
                
                # Check max probabilities
                max_probs = st_log_probs.max(dim=-1)[0]  # [T, B]
                avg_max_prob = max_probs.mean().item()
                print(f"   Avg max prob: {avg_max_prob:.4f}")
                print(f"   Min max prob: {max_probs.min().item():.4f}")
                print(f"   Max max prob: {max_probs.max().item():.4f}")
            
            # Check outputs
            if 'asr_logits' in output and output['asr_logits'] is not None:
                asr_shape = output['asr_logits'].shape
                print(f"   ASR logits: {asr_shape}")
            else:
                print(f"   ASR logits: None")
            
            if 'st_logits' in output and output['st_logits'] is not None:
                st_shape = output['st_logits'].shape
                print(f"   ST logits: {st_shape}")
            else:
                print(f"   ST logits: None")
            
            if 'mt_logits' in output and output['mt_logits'] is not None:
                mt_shape = output['mt_logits'].shape
                print(f"   MT logits: {mt_shape}")
            else:
                print(f"   MT logits: None")
            
            if 'unit_logits' in output and output['unit_logits'] is not None:
                unit_shape = output['unit_logits'].shape
                print(f"   Unit logits: {unit_shape}")
            else:
                print(f"   Unit logits: None")
            
            if 'units' in output and output['units'] is not None:
                units_shape = output['units'].shape
                print(f"   Units: {units_shape}")
            else:
                print(f"   Units: None")
            
            if 'waveform' in output and output['waveform'] is not None:
                wav_shape = output['waveform'].shape
                wav_duration = wav_shape[-1] / sample_rate
                print(f"   Waveform: {wav_shape} ({wav_duration:.2f}s)")
            else:
                print(f"   Waveform: None (vocoder not enabled or failed)")
            
            print("\n" + "="*70)
            print("✅ Inference test completed successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EchoStream inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/echostream_config.mini.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_mini_units_v4/checkpoint_best.pt",
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Audio duration in seconds (for random audio)"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Path to real audio file (if provided, uses this instead of random audio)"
    )
    
    args = parser.parse_args()
    
    success = test_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        use_random_audio=args.audio_path is None,
        audio_path=args.audio_path,
        audio_duration_sec=args.duration,
    )
    
    sys.exit(0 if success else 1)

