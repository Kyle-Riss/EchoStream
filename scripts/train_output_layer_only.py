#!/usr/bin/env python3
"""
Train only the ST CTC Output Linear Layer (ctc_proj)
- Freeze encoder and all other layers
- Re-initialize only st_ctc_decoder.ctc_proj
- Higher LR (3e-5 ~ 1e-4)
- 5-10 epochs on 2,400 samples
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
import yaml
import os

from scripts.train import MultiTaskLoss, build_echostream_model, EchoStreamConfig
from datasets import S2STManifestDataset, collate_s2st_batches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint to load')
    parser.add_argument('--train-manifest', type=str, required=True)
    parser.add_argument('--dev-manifest', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='checkpoints_output_layer_retrain')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 Output Linear Layer Only - Re-training")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load model config
    config = EchoStreamConfig()
    config.use_simple_encoder = True
    config.encoder_embed_dim = 128
    config.encoder_layers = 4
    config.encoder_ffn_embed_dim = 512
    config.decoder_embed_dim = 128
    config.mt_decoder_layers = 2
    config.unit_decoder_layers = 2
    config.st_decoder_layers = 2
    config.dropout = 0.1
    
    # Build model
    model = build_echostream_model(config)
    
    # Monkey patch model.forward to skip unit_decoder (we only train ST)
    original_forward = model.forward
    def forward_without_unit(src_tokens, src_lengths, prev_output_tokens=None, target_lengths=None, **kwargs):
        # Call original forward but catch unit_decoder error
        try:
            return original_forward(src_tokens, src_lengths, prev_output_tokens, target_lengths, **kwargs)
        except RuntimeError as e:
            if 'size of tensor a' in str(e) and 'size of tensor b' in str(e):
                # This is the unit_decoder positional encoding error
                # Return output without unit_decoder
                # We'll manually construct the output dict
                encoder_out = model.encoder(src_tokens, src_lengths)
                encoder_hidden = encoder_out['encoder_out'][0]
                encoder_padding_mask = encoder_out['encoder_padding_mask'][0] if encoder_out['encoder_padding_mask'] else None
                
                asr_out = model.asr_ctc_decoder(encoder_out=encoder_hidden, encoder_padding_mask=encoder_padding_mask)
                st_out = model.st_ctc_decoder(encoder_out=encoder_hidden, encoder_padding_mask=encoder_padding_mask)
                
                return {
                    'asr_log_probs': asr_out['log_probs'],
                    'st_log_probs': st_out['log_probs'],
                    'mt_logits': None,
                    'unit_log_probs': None,
                }
            else:
                raise
    model.forward = forward_without_unit
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Filter vocoder keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder.')}
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # ============================================
    # FREEZE everything except ST CTC output layer
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("🔒 Freezing all layers except st_ctc_decoder.ctc_proj")
    logger.info("="*70)
    
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if 'st_ctc_decoder.ctc_proj' in name:
            # Keep this trainable
            param.requires_grad = True
            trainable_params.append((name, param.numel()))
            logger.info(f"✅ TRAINABLE: {name} {list(param.shape)}")
        else:
            # Freeze everything else
            param.requires_grad = False
            frozen_params.append((name, param.numel()))
    
    total_trainable = sum(p[1] for p in trainable_params)
    total_frozen = sum(p[1] for p in frozen_params)
    
    logger.info(f"\n📊 Trainable params: {total_trainable:,}")
    logger.info(f"📊 Frozen params: {total_frozen:,}")
    logger.info(f"📊 Trainable ratio: {total_trainable/(total_trainable+total_frozen)*100:.2f}%")
    
    # ============================================
    # RE-INITIALIZE ST CTC output layer
    # ============================================
    logger.info("\n" + "="*70)
    logger.info("🔄 Re-initializing st_ctc_decoder.ctc_proj")
    logger.info("="*70)
    
    with torch.no_grad():
        # Re-init weights
        nn.init.xavier_uniform_(model.st_ctc_decoder.ctc_proj.weight)
        if model.st_ctc_decoder.ctc_proj.bias is not None:
            nn.init.constant_(model.st_ctc_decoder.ctc_proj.bias, 0.0)
    
    logger.info("✅ Output layer re-initialized")
    
    model = model.to(device)
    
    # ============================================
    # Optimizer (only trainable params)
    # ============================================
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_param_list, lr=args.lr)
    
    logger.info(f"\n✅ Optimizer: Adam (lr={args.lr})")
    logger.info(f"   Trainable params: {len(trainable_param_list)}")
    
    # ============================================
    # Loss function
    # ============================================
    loss_fn = MultiTaskLoss(
        asr_weight=0.0,
        st_weight=1.0,  # ST-only
        mt_weight=0.0,
        unit_weight=0.0,
    )
    
    # ============================================
    # Dataset
    # ============================================
    train_dataset = S2STManifestDataset(
        manifest_path=args.train_manifest,
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    dev_dataset = S2STManifestDataset(
        manifest_path=args.dev_manifest,
        data_root='data',
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path='data/tgt_unigram6000/spm_unigram_en_sp_format.txt',
        text_level='word',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_s2st_batches,
        num_workers=args.num_workers,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_s2st_batches,
        num_workers=args.num_workers,
    )
    
    logger.info(f"\n✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Dev samples: {len(dev_dataset)}")
    
    # ============================================
    # Training loop
    # ============================================
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_dev_loss = float('inf')
    start_epoch = 0
    
    logger.info("\n" + "="*70)
    logger.info("🚀 Starting training...")
    logger.info("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            speech = batch['speech'].to(device)
            speech_lengths = batch['speech_lengths'].to(device)
            target_text = batch['target_text'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            
            # Forward
            output = model(
                src_tokens=speech,
                src_lengths=speech_lengths,
                prev_output_tokens=None,
                target_lengths=target_lengths,
            )
            
            # Loss
            model_output = {
                'st_log_probs': output['st_log_probs'],
            }
            target = {
                'target_text': target_text,
                'target_lengths': target_lengths,
            }
            total_loss, loss_dict = loss_fn(model_output, target)
            
            loss = total_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Dev
        model.eval()
        dev_loss = 0.0
        dev_batches = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                speech = batch['speech'].to(device)
                speech_lengths = batch['speech_lengths'].to(device)
                target_text = batch['target_text'].to(device)
                target_lengths = batch['target_lengths'].to(device)
                
                output = model(
                    src_tokens=speech,
                    src_lengths=speech_lengths,
                    prev_output_tokens=None,
                    target_lengths=target_lengths,
                )
                
                model_output = {
                    'st_log_probs': output['st_log_probs'],
                }
                target = {
                    'target_text': target_text,
                    'target_lengths': target_lengths,
                }
                total_loss, loss_dict = loss_fn(model_output, target)
                
                dev_loss += total_loss.item()
                dev_batches += 1
        
        avg_dev_loss = dev_loss / dev_batches
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Dev Loss: {avg_dev_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'dev_loss': avg_dev_loss,
        }, checkpoint_path)
        
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            best_path = save_dir / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'dev_loss': avg_dev_loss,
            }, best_path)
            logger.info(f"  ✅ Best model saved (dev_loss={best_dev_loss:.4f})")
    
    logger.info("\n" + "="*70)
    logger.info("🎉 Training completed!")
    logger.info("="*70)


if __name__ == '__main__':
    main()

