#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_wpb.py

- StreamSpeech encoder 출력(enc_out)을 사용해서
  WordPriorBooster(WPB)를 학습하는 최소 예제.

실제 EchoStream 구조에 맞게 통합:
- STCTCWithLM 모델 사용
- S2STManifestDataset 사용
- encoder 출력을 실제로 가져와서 WPB 학습
"""
import json
from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches
from models.word_prior_booster import WordPriorBooster
from models.st_ctc_lm import STCTCWithLM
from utils.vocab_utils import load_word_list, make_multilabel_targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_st_model(checkpoint_path: str, vocab_size: int, encoder_dim: int = 128, device="cpu"):
    """
    STCTCWithLM 모델 로드 (encoder만 사용, freeze)
    """
    model = STCTCWithLM(
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        encoder_layers=4,
        decoder_dim=encoder_dim,
        decoder_layers=3,
        dropout=0.1,
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        # encoder만 로드
        encoder_state = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        if encoder_state:
            model.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(f"Loaded encoder from {checkpoint_path}")
    
    model.to(device)
    model.eval()  # encoder freeze
    
    # encoder만 사용하도록 설정
    for p in model.encoder.parameters():
        p.requires_grad = False
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Train manifest TSV")
    parser.add_argument("--word-list", type=str, required=True, help="word_list.json 경로")
    parser.add_argument("--spm-model", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    parser.add_argument("--save-dir", type=str, default="checkpoints_wpb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--base-checkpoint", type=str, default=None, help="기존 ST 모델 checkpoint (encoder만 사용)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) vocab/word_list 로드
    vocab_word = load_word_list(args.word_list)
    num_words = len(vocab_word)
    logger.info(f"Loaded word vocab: {num_words} words")

    # SPM tokenizer로 vocab_size 확인
    from datasets.s2st_dataset import SPMTokenizer
    spm_tokenizer = SPMTokenizer(args.spm_model)
    vocab_size = len(spm_tokenizer)

    # 2) base model 로드 (encoder만 사용)
    base_model = load_st_model(args.base_checkpoint, vocab_size, args.encoder_dim, device)
    logger.info("Base ST model loaded (encoder frozen)")

    # 3) WPB 준비
    wpb = WordPriorBooster(enc_dim=args.encoder_dim, num_words=num_words).to(device)
    optimizer = torch.optim.Adam(wpb.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # 4) 데이터
    dataset = S2STManifestDataset(
        manifest_path=args.manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.spm_model,
        text_level="word",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_s2st_batches,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    # 5) 학습 루프
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        wpb.train()
        total_loss = 0.0
        steps = 0

        for batch in loader:
            speech = batch["speech"].to(device)  # [B, T, 80]
            speech_lengths = batch["speech_lengths"].to(device)  # [B]
            
            # 텍스트 리스트 추출 (batch에서)
            texts = []
            if "tgt_text" in batch:
                # tgt_text가 tensor인 경우
                if isinstance(batch["tgt_text"], torch.Tensor):
                    tgt_tokens = batch["tgt_text"]
                    tgt_lengths = batch["target_lengths"]
                    for i in range(tgt_tokens.size(0)):
                        tokens = tgt_tokens[i][:tgt_lengths[i]-1].cpu().tolist()
                        texts.append(spm_tokenizer.sp.decode(tokens))
                else:
                    texts = batch["tgt_text"]
            else:
                # fallback: dataset에서 직접 가져오기
                batch_indices = batch.get("indices", list(range(len(batch["speech"]))))
                texts = [dataset[idx]["tgt_text"] for idx in batch_indices]

            with torch.no_grad():
                # 실제 encoder 출력 가져오기
                enc_out_dict = base_model.encoder(speech, speech_lengths)
                enc_out = enc_out_dict["encoder_out"][0]  # [T, B, D]
                enc_out = enc_out.transpose(0, 1)  # [B, T, D]

            # WPB forward (배치 전체를 한 번에 처리, padding은 mean pooling에서 자동 처리됨)
            logits_prior = wpb(enc_out)  # [B, num_words]

            # 멀티레이블 타겟 생성
            targets = make_multilabel_targets(texts, vocab_word).to(device)  # [B, num_words]

            loss = criterion(logits_prior, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 50 == 0:
                logger.info(f"Epoch {epoch} step {steps} loss {total_loss/steps:.4f}")

        avg_loss = total_loss / max(1, steps)
        logger.info(f"Epoch {epoch} done, avg_loss={avg_loss:.4f}")

        # 체크포인트 저장
        torch.save(
            {
                "model_state": wpb.state_dict(),
                "enc_dim": args.encoder_dim,
                "vocab_word": vocab_word,
                "num_words": num_words,
                "config": vars(args),
            },
            save_dir / f"checkpoint_epoch_{epoch}.pt",
        )
        logger.info(f"Saved {save_dir}/checkpoint_epoch_{epoch}.pt")

    logger.info(f"[+] Training complete. Final checkpoint: {save_dir}/checkpoint_epoch_{args.epochs}.pt")


if __name__ == "__main__":
    main()
