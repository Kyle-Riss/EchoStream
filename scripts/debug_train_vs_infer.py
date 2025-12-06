"""
Debug script to compare training path vs inference path
"""
import sys
sys.path.insert(0, '.')

import torch
import yaml
from pathlib import Path
from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.echostream_model import build_echostream_model, EchoStreamConfig

def main():
    print("="*70)
    print("Training Path vs Inference Path Comparison")
    print("="*70)
    
    # Load config
    config_path = 'configs/echostream_config.mini.yaml'
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Build config (same as train.py)
    config_overrides = {}
    encoder_cfg = config_dict.get('encoder', {})
    if encoder_cfg:
        if 'embed_dim' in encoder_cfg:
            config_overrides['encoder_embed_dim'] = encoder_cfg['embed_dim']
        if 'layers' in encoder_cfg:
            config_overrides['encoder_layers'] = encoder_cfg['layers']
        if 'ffn_embed_dim' in encoder_cfg:
            config_overrides['encoder_ffn_embed_dim'] = encoder_cfg['ffn_embed_dim']
    
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
    
    st_decoder_cfg = config_dict.get('st_decoder', {})
    if st_decoder_cfg and 'layers' in st_decoder_cfg:
        config_overrides['st_decoder_layers'] = st_decoder_cfg['layers']
    
    config = EchoStreamConfig.from_dict(config_overrides)
    
    # Build model
    model = build_echostream_model(config)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints_spm_fixed/checkpoint_best.pt'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"Model vocab size: {model.st_ctc_decoder.vocab_size}")
    print(f"ST CTC projection shape: {model.st_ctc_decoder.ctc_proj.weight.shape}")
    
    # Load one sample from training dataset
    data_cfg = config_dict.get('data', {})
    dataset = S2STManifestDataset(
        manifest_path='data/train_sampled.units.streamspeech_format_final.tsv',
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_with_special.txt',
        text_level='word',
    )
    
    print(f"\nDataset vocab size: {len(dataset.tgt_tokenizer.id_to_token) if hasattr(dataset, 'tgt_tokenizer') else 'N/A'}")
    
    # Get one sample
    sample = dataset[0]
    print(f"\nSample info:")
    print(f"  Speech shape: {sample['speech'].shape}")
    print(f"  Target tokens: {sample['tgt_tokens'][:20]}")
    print(f"  Target contains blank(0): {(sample['tgt_tokens'] == 0).any().item()}")
    
    # Create batch
    batch = collate_s2st_batches([sample])
    
    print("\n" + "="*70)
    print("TEST 1: Training Path (model.train())")
    print("="*70)
    
    # CTC blank index = 1 (PAD token in SentencePiece)
    BLANK_ID = 1
    
    model.train()
    with torch.no_grad():
        output_train = model(
            src_tokens=batch['speech'],
            src_lengths=batch['speech_lengths'],
            prev_output_tokens=None,
            target_lengths=batch.get('target_lengths'),
        )
        
        st_log_probs_train = output_train['st_log_probs']  # [T, B, V]
        st_tokens_train = st_log_probs_train.argmax(dim=-1)  # [T, B]
        
        blank_count_train = (st_tokens_train == BLANK_ID).sum().item()
        total_train = st_tokens_train.numel()
        blank_ratio_train = blank_count_train / total_train
        
        print(f"ST log_probs shape: {st_log_probs_train.shape}")
        print(f"Blank ({BLANK_ID}) count: {blank_count_train}/{total_train} ({blank_ratio_train:.1%})")
        print(f"First 30 tokens: {st_tokens_train[:30, 0].tolist()}")
        print(f"Unique tokens: {torch.unique(st_tokens_train).tolist()[:20]}")
        print(f"Blank({BLANK_ID}) log_prob: {st_log_probs_train[:, :, BLANK_ID].mean():.4f}")
        print(f"Non-blank log_prob: {st_log_probs_train[:, :, [i for i in range(st_log_probs_train.size(-1)) if i != BLANK_ID]].mean():.4f}")
    
    print("\n" + "="*70)
    print("TEST 2: Inference Path (model.eval())")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        output_eval = model(
            src_tokens=batch['speech'],
            src_lengths=batch['speech_lengths'],
            prev_output_tokens=None,
            target_lengths=batch.get('target_lengths'),
        )
        
        st_log_probs_eval = output_eval['st_log_probs']  # [T, B, V]
        st_tokens_eval = st_log_probs_eval.argmax(dim=-1)  # [T, B]
        
        blank_count_eval = (st_tokens_eval == BLANK_ID).sum().item()
        total_eval = st_tokens_eval.numel()
        blank_ratio_eval = blank_count_eval / total_eval
        
        print(f"ST log_probs shape: {st_log_probs_eval.shape}")
        print(f"Blank ({BLANK_ID}) count: {blank_count_eval}/{total_eval} ({blank_ratio_eval:.1%})")
        print(f"First 30 tokens: {st_tokens_eval[:30, 0].tolist()}")
        print(f"Unique tokens: {torch.unique(st_tokens_eval).tolist()[:20]}")
        print(f"Blank({BLANK_ID}) log_prob: {st_log_probs_eval[:, :, BLANK_ID].mean():.4f}")
        print(f"Non-blank log_prob: {st_log_probs_eval[:, :, [i for i in range(st_log_probs_eval.size(-1)) if i != BLANK_ID]].mean():.4f}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Blank ratio difference: {abs(blank_ratio_train - blank_ratio_eval):.1%}")
    print(f"Log probs identical: {torch.allclose(st_log_probs_train, st_log_probs_eval, atol=1e-6)}")
    
    if not torch.allclose(st_log_probs_train, st_log_probs_eval, atol=1e-6):
        print("\n⚠️  WARNING: Training and inference paths produce DIFFERENT outputs!")
        print("   This explains the 100% blank issue in inference.")
    else:
        print("\n✅ Training and inference paths are identical.")
        if blank_ratio_eval > 0.9:
            print("   But both produce 100% blank - this is a model learning issue.")

if __name__ == "__main__":
    main()

