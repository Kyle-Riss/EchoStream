#!/usr/bin/env python3
"""
Unit Decoder training script.

Trains Unit Decoder to convert text hidden states to discrete speech units.
Can train Unit Decoder alone (with frozen ST model) or jointly with ST model.
"""
import sys
import os
from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches, SPMTokenizer
from models.st_ctc_lm import STCTCWithLM
from models.decoders.unit_decoder import CTCTransformerUnitDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def freeze_st_model(model: STCTCWithLM, freeze: bool):
    """Freeze or unfreeze ST model parameters."""
    for name, p in model.named_parameters():
        if name.startswith("unit_decoder"):
            # Unit Decoder always trainable
            p.requires_grad = True
            continue
        p.requires_grad = not freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--dev-manifest", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    parser.add_argument("--checkpoint-st", type=str, required=True, help="ST model checkpoint path")
    parser.add_argument("--save-dir", type=str, default="checkpoints_unit_decoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-st", action="store_true", help="Freeze ST model, train only Unit Decoder")
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--unit-decoder-layers", type=int, default=6)
    parser.add_argument("--unit-decoder-heads", type=int, default=4)
    parser.add_argument("--unit-decoder-ffn-dim", type=int, default=1024)
    parser.add_argument("--num-units", type=int, default=1000)
    parser.add_argument("--ctc-upsample-ratio", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-mt-decoder", action="store_true", help="Use MT Decoder for text hidden states")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load tokenizer
    spm = SPMTokenizer(args.vocab_path)
    vocab_size = len(spm)
    pad_idx = spm.pad_id
    bos_idx = spm.bos_id
    eos_idx = spm.eos_id
    logger.info(f"Vocab size={vocab_size} | pad={pad_idx} bos={bos_idx} eos={eos_idx}")

    # Load checkpoint first to get config
    checkpoint = torch.load(args.checkpoint_st, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Get decoder_ffn_dim from checkpoint if available
    decoder_ffn_dim = args.decoder_ffn_dim if hasattr(args, 'decoder_ffn_dim') else 512
    if 'decoder.decoder.layers.0.linear1.weight' in state_dict:
        # Infer decoder_ffn_dim from checkpoint
        decoder_ffn_dim = state_dict['decoder.decoder.layers.0.linear1.weight'].shape[0]
        logger.info(f"Using decoder_ffn_dim={decoder_ffn_dim} from checkpoint")
    
    # Load ST model
    logger.info(f"Loading ST model from {args.checkpoint_st}")
    st_model = STCTCWithLM(
        vocab_size=vocab_size,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        use_mt_decoder=args.use_mt_decoder,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    ).to(device)
    
    # Load ST model weights (excluding unit_decoder if exists)
    st_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('unit_decoder')}
    st_model.load_state_dict(st_state_dict, strict=False)
    st_model.eval()  # ST model in eval mode
    
    # Initialize Unit Decoder
    unit_decoder = CTCTransformerUnitDecoder(
        input_dim=args.decoder_dim,
        embed_dim=args.decoder_dim,
        num_layers=args.unit_decoder_layers,
        num_heads=args.unit_decoder_heads,
        ffn_embed_dim=args.unit_decoder_ffn_dim,
        num_units=args.num_units,
        ctc_upsample_ratio=args.ctc_upsample_ratio,
        dropout=args.dropout,
    ).to(device)
    
    # Attach Unit Decoder to ST model
    st_model.unit_decoder = unit_decoder
    
    # Freeze ST model if requested
    if args.freeze_st:
        freeze_st_model(st_model, freeze=True)
        logger.info("ST model frozen, training only Unit Decoder")
    else:
        freeze_st_model(st_model, freeze=False)
        logger.info("Training ST model and Unit Decoder jointly")

    # Data
    train_dataset = S2STManifestDataset(
        manifest_path=args.train_manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
        load_tgt_units=True,  # Load target units for training
    )
    dev_dataset = S2STManifestDataset(
        manifest_path=args.dev_manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
        load_tgt_units=True,
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
    logger.info(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss function
    unit_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

    # Optimizer
    if args.freeze_st:
        optimizer = torch.optim.Adam(unit_decoder.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(st_model.parameters(), lr=args.lr)
    
    scaler = GradScaler()

    def run_epoch(epoch: int, is_train: bool):
        if is_train:
            st_model.train()
            unit_decoder.train()
            loader = train_loader
        else:
            st_model.eval()
            unit_decoder.eval()
            loader = dev_loader

        total_loss = 0.0
        total_samples = 0
        total_units = 0

        for batch_idx, batch in enumerate(loader):
            # Check batch keys
            if "speech" in batch:
                src_feats = batch["speech"].to(device)  # [B, T, F]
                src_lengths = batch["speech_lengths"].to(device)  # [B]
            else:
                src_feats = batch["src_feats"].to(device)  # [B, T, F]
                src_lengths = batch["src_lengths"].to(device)  # [B]
            
            if "target_text" in batch:
                tgt_tokens = batch["target_text"].to(device)  # [B, T_tgt]
                tgt_lengths = batch["target_lengths"].to(device)  # [B]
            else:
                tgt_tokens = batch["tgt_tokens"].to(device)  # [B, T_tgt]
                tgt_lengths = batch["tgt_lengths"].to(device)  # [B]
            
            tgt_units = batch.get("tgt_units", None)  # [B, T_units] or None

            if tgt_units is None:
                logger.warning(f"Batch {batch_idx}: tgt_units not found, skipping")
                continue

            tgt_units = tgt_units.to(device)  # [B, T_units]
            tgt_units_lengths = batch.get("tgt_units_lengths", None)
            if tgt_units_lengths is not None:
                tgt_units_lengths = tgt_units_lengths.to(device)

            optimizer.zero_grad()

            with autocast():
                # Forward through ST model
                st_output = st_model(
                    src_feats=src_feats,
                    src_lengths=src_lengths,
                    tgt_tokens=tgt_tokens,
                    tgt_lengths=tgt_lengths,
                    compute_loss=False,
                )

                # Get text hidden states for Unit Decoder
                if args.use_mt_decoder and st_model.mt_decoder is not None:
                    # Use MT Decoder output
                    encoder_out_dict = {
                        'encoder_out': [st_output['encoder_out']],
                        'encoder_padding_mask': [st_output['encoder_padding_mask']] if st_output['encoder_padding_mask'] is not None else None,
                    }
                    mt_out = st_model.mt_decoder(
                        prev_output_tokens=tgt_tokens,
                        encoder_out=encoder_out_dict,
                    )
                    text_hidden = mt_out.get('decoder_out', st_output['encoder_out'])  # [T, B, D]
                else:
                    # Use encoder output directly
                    text_hidden = st_output['encoder_out']  # [T, B, D]

                # Forward through Unit Decoder
                unit_out = unit_decoder(
                    text_hidden=text_hidden,
                    text_padding_mask=st_output['encoder_padding_mask'],
                )

                # Compute loss
                unit_log_probs = unit_out['log_probs']  # [B, T_unit, num_units]
                B, T_unit, V = unit_log_probs.shape

                # Prepare targets: [B, T_unit]
                # If tgt_units_lengths provided, use them; otherwise assume full length
                if tgt_units_lengths is not None:
                    # Pad or truncate tgt_units to match T_unit
                    tgt_units_padded = torch.full((B, T_unit), -1, dtype=torch.long, device=device)
                    for b in range(B):
                        t_len = min(int(tgt_units_lengths[b].item()), T_unit)
                        if t_len > 0:
                            tgt_units_padded[b, :t_len] = tgt_units[b, :t_len]
                    targets = tgt_units_padded
                else:
                    # Truncate or pad tgt_units to T_unit
                    tgt_units_padded = torch.full((B, T_unit), -1, dtype=torch.long, device=device)
                    for b in range(B):
                        t_len = min(tgt_units[b].size(0), T_unit)
                        if t_len > 0:
                            tgt_units_padded[b, :t_len] = tgt_units[b, :t_len]
                    targets = tgt_units_padded

                # Reshape for loss computation
                unit_log_probs_flat = unit_log_probs.reshape(B * T_unit, V)  # [B*T_unit, V]
                targets_flat = targets.reshape(B * T_unit)  # [B*T_unit]

                loss = unit_loss_fn(unit_log_probs_flat, targets_flat)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * B
            total_samples += B
            total_units += (targets != -1).sum().item()

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Epoch {epoch} {'Train' if is_train else 'Dev'} | "
                    f"Batch {batch_idx+1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info(
            f"Epoch {epoch} {'Train' if is_train else 'Dev'} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Samples: {total_samples} | Units: {total_units}"
        )
        return avg_loss

    best_dev_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(epoch, is_train=True)
        dev_loss = run_epoch(epoch, is_train=False)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': st_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
        }
        torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(checkpoint, save_dir / "checkpoint_best.pt")
            logger.info(f"✅ Saved best checkpoint (dev_loss={dev_loss:.4f})")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

