#!/usr/bin/env python3
"""
CTC + Lightweight Transformer Decoder (conditional LM) training.

Two-stage schedule:
  1) LM warmup: freeze encoder + CTC head, train decoder only (lambda_ctc=0, lambda_xe=1)
  2) Joint finetune: unfreeze all, lambda_ctc/lambda_xe set by args.
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

sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches, SPMTokenizer
from models.st_ctc_lm import STCTCWithLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def freeze_encoder_and_ctc(model: STCTCWithLM, freeze: bool):
    for name, p in model.named_parameters():
        if name.startswith("decoder."):
            # decoder always trainable
            p.requires_grad = True
            continue
        if name.startswith("ctc_head") or name.startswith("encoder"):
            p.requires_grad = not freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--dev-manifest", type=str, required=True)
    # SentencePiece model path (binary .model)
    parser.add_argument("--vocab-path", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    parser.add_argument("--save-dir", type=str, default="checkpoints_st_lm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-ctc", type=float, default=0.3)
    parser.add_argument("--lambda-xe", type=float, default=0.7)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--decoder-ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints_st_lm_vocab5000/checkpoint_epoch_4.pt)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load tokenizer to ensure pad/bos/eos indices match vocab
    spm = SPMTokenizer(args.vocab_path)
    vocab_size = len(spm)
    pad_idx = spm.pad_id
    bos_idx = spm.bos_id
    eos_idx = spm.eos_id
    logger.info(
        f"Vocab size={vocab_size} | pad={pad_idx} bos={bos_idx} eos={eos_idx} (from {args.vocab_path})"
    )

    model = STCTCWithLM(
        vocab_size=vocab_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        dropout=args.dropout,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        lambda_ctc=args.lambda_ctc,
        lambda_xe=args.lambda_xe,
    ).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                start_epoch = ckpt.get("epoch", 1) + 1
                logger.info(f"Resumed from epoch {ckpt.get('epoch', 1)}, starting from epoch {start_epoch}")
            else:
                model.load_state_dict(ckpt, strict=False)
                logger.info("Loaded model state (no epoch info, starting from epoch 1)")
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}, starting from scratch")

    # Data
    train_dataset = S2STManifestDataset(
        manifest_path=args.train_manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
    )
    dev_dataset = S2STManifestDataset(
        manifest_path=args.dev_manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
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

    def run_epoch(epoch: int, warmup: bool):
        model.train()
        if warmup:
            freeze_encoder_and_ctc(model, freeze=True)
            lambda_ctc = 0.0
            lambda_xe = 1.0
        else:
            freeze_encoder_and_ctc(model, freeze=False)
            lambda_ctc = args.lambda_ctc
            lambda_xe = args.lambda_xe

        # Recreate optimizer per phase to respect requires_grad flags
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        scaler = GradScaler()

        total_loss = 0.0
        total_ctc = 0.0
        total_xe = 0.0
        steps = 0

        for batch in train_loader:
            src = batch["speech"].to(device)
            src_len = batch["speech_lengths"].to(device)
            tgt = batch["target_text"].to(device)
            tgt_len = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            with autocast():
                out = model(src, src_len, tgt_tokens=tgt, tgt_lengths=tgt_len, compute_loss=True)
                # Override lambdas for warmup / joint
                loss = lambda_ctc * out["ctc_loss"] + lambda_xe * out["xe_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ctc += out["ctc_loss"].item()
            total_xe += out["xe_loss"].item()
            steps += 1

            if steps % 50 == 0:
                logger.info(
                    f"[Epoch {epoch}] step {steps} "
                    f"loss={total_loss/steps:.3f} "
                    f"ctc={total_ctc/steps:.3f} "
                    f"xe={total_xe/steps:.3f} "
                    f"warmup={warmup}"
                )

        logger.info(
            f"[Epoch {epoch}] train done | "
            f"loss={total_loss/steps:.3f} ctc={total_ctc/steps:.3f} xe={total_xe/steps:.3f} | warmup={warmup}"
        )

    @torch.no_grad()
    def evaluate(epoch: int):
        model.eval()
        total_loss = 0.0
        total_ctc = 0.0
        total_xe = 0.0
        steps = 0
        for batch in dev_loader:
            src = batch["speech"].to(device)
            src_len = batch["speech_lengths"].to(device)
            tgt = batch["target_text"].to(device)
            tgt_len = batch["target_lengths"].to(device)
            out = model(src, src_len, tgt_tokens=tgt, tgt_lengths=tgt_len, compute_loss=True)
            loss = args.lambda_ctc * out["ctc_loss"] + args.lambda_xe * out["xe_loss"]
            total_loss += loss.item()
            total_ctc += out["ctc_loss"].item()
            total_xe += out["xe_loss"].item()
            steps += 1
        if steps > 0:
            logger.info(
                f"[Epoch {epoch}] dev | loss={total_loss/steps:.3f} "
                f"ctc={total_ctc/steps:.3f} xe={total_xe/steps:.3f}"
            )

    for epoch in range(start_epoch, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs
        run_epoch(epoch, warmup=warmup)
        evaluate(epoch)

        ckpt = {
            "model": model.state_dict(),
            "config": vars(args),
            "epoch": epoch,
        }
        torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch}.pt")
        logger.info(f"Saved checkpoint: {save_dir}/checkpoint_epoch_{epoch}.pt")


if __name__ == "__main__":
    main()

