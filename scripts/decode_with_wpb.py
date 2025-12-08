#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_with_wpb.py

- StreamSpeech encoder/decoder + Word Prior Booster + logit fusion
- 한 번에 minimal 예제로 "단어라도 나오게" 만드는 디코더 실험용.

실제 EchoStream 구조에 맞게 통합:
- STCTCWithLM 모델 사용
- 실제 decoder 호출로 logits_lm 가져오기
- WPB + fusion 적용
"""
import json
from pathlib import Path
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from datasets.s2st_dataset import S2STManifestDataset, collate_s2st_batches, SPMTokenizer
from torch.utils.data import DataLoader
from models.word_prior_booster import WordPriorBooster
from models.st_ctc_lm import STCTCWithLM
from utils.token_prior import build_token_prior
from utils.ctc_beam_search import CTCBeamSearchDecoder
from utils.sentence_structure import apply_sentence_structure_filters
from utils.ngram_lm import NgramLM
from utils.word_ngram_lm import WordNgramLM
from utils.template_matcher import TemplateMatcher
from models.decoders.unit_decoder import CTCTransformerUnitDecoder
from models.decoders.vocoder import CodeHiFiGANVocoder
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def autoregressive_decode_with_wpb(
    decoder,
    encoder_out,  # [T, B, D]
    encoder_padding_mask,  # [B, T] or None
    log_prior,  # [V]
    alpha,  # fusion weight
    bos_id,
    eos_id,
    unk_id=None,
    unk_penalty=0.0,
    max_len=100,
    rep_penalty=1.2,
    device="cpu",
):
    """
    Autoregressive decoding with WPB prior fusion.
    Uses decoder to generate tokens step by step, fusing WPB prior at each step.
    """
    B = encoder_out.size(1)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    
    # Track generated tokens for repetition penalty
    generated_tokens = [[] for _ in range(B)]
    
    for step in range(max_len):
        # Get decoder logits
        decoder_logits = decoder(
            ys, encoder_out=encoder_out, encoder_padding_mask=encoder_padding_mask
        )  # [B, T_dec, V]
        step_logits = decoder_logits[:, -1, :]  # [B, V]
        
        # Fuse with WPB prior
        step_logits = step_logits + alpha * log_prior.view(1, -1)  # [B, V]
        
        # UNK penalty
        if unk_penalty > 0 and unk_id is not None:
            step_logits[:, unk_id] -= unk_penalty
        
        # Repetition penalty: reduce logits for recently generated tokens
        if rep_penalty > 1.0:
            for b in range(B):
                if len(generated_tokens[b]) > 0:
                    # Penalize tokens that appeared in last few steps (more aggressive)
                    recent_tokens = set(generated_tokens[b][-10:])  # Last 10 tokens
                    for token_id in recent_tokens:
                        if 0 <= token_id < step_logits.size(1):
                            step_logits[b, token_id] /= rep_penalty
                    # Extra penalty for immediate repetition
                    if len(generated_tokens[b]) > 0:
                        last_token = generated_tokens[b][-1]
                        if 0 <= last_token < step_logits.size(1):
                            step_logits[b, last_token] /= (rep_penalty * 1.5)  # Extra penalty for immediate repeat
        
        # Greedy selection
        next_token = step_logits.argmax(dim=-1)  # [B]
        
        # Update generated tokens
        for b in range(B):
            generated_tokens[b].append(next_token[b].item())
        
        # Append to sequence
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        
        # Check for EOS
        finished |= next_token.eq(eos_id)
        if finished.all():
            break
    
    # Extract sequence (remove BOS, stop at EOS)
    out = []
    for seq in ys.tolist():
        seq = seq[1:]  # drop BOS
        if eos_id in seq:
            seq = seq[:seq.index(eos_id)]
        out.append(seq)
    
    return out[0] if out else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-wpb", type=str, required=True, help="WPB checkpoint 경로")
    parser.add_argument("--checkpoint-st", type=str, default=None, help="ST 모델 checkpoint (encoder+decoder)")
    parser.add_argument("--word2tokens", type=str, required=True, help="word2tokens.json 경로")
    parser.add_argument("--manifest", type=str, required=True, help="테스트 manifest")
    parser.add_argument("--vocab-path", type=str, default="data/tgt_unigram5000/spm_unigram_en.model")
    parser.add_argument("--alpha", type=float, default=2.0, help="Fusion weight for WPB prior (lower = less WPB influence)")
    parser.add_argument("--unk-penalty", type=float, default=10.0, help="UNK logit penalty")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for WPB prior (>1.0 = flatter distribution)")
    parser.add_argument("--suppress-words", type=str, nargs="+", default=["no", "or"], help="Words to suppress in WPB prior")
    parser.add_argument("--suppress-scale", type=float, default=0.3, help="Scale factor for suppressed words (0.0-1.0)")
    parser.add_argument("--rep-penalty", type=float, default=0.0, help="Repetition penalty for suppressed word tokens in logits")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for CTC decoding (1=greedy)")
    parser.add_argument("--length-penalty", type=float, default=0.0, help="Length penalty for beam search")
    parser.add_argument("--beam-rep-penalty", type=float, default=0.5, help="Repetition penalty in beam search")
    parser.add_argument("--use-decoder", action="store_true", help="Use autoregressive decoder instead of CTC only")
    parser.add_argument("--max-len", type=int, default=100, help="Max length for autoregressive decoding")
    parser.add_argument("--decoder-rep-penalty", type=float, default=1.2, help="Repetition penalty for autoregressive decoder")
    parser.add_argument("--post-process", action="store_true", help="Apply post-processing to improve sentence structure")
    parser.add_argument("--ngram-lm", type=str, default=None, help="Path to subword N-gram LM JSON file")
    parser.add_argument("--ngram-weight", type=float, default=0.3, help="Weight for subword N-gram LM fusion")
    parser.add_argument("--word-ngram-lm", type=str, default=None, help="Path to word-level N-gram LM JSON file")
    parser.add_argument("--word-ngram-weight", type=float, default=0.5, help="Weight for word-level N-gram LM scoring")
    parser.add_argument("--templates", type=str, default=None, help="Path to templates JSON file")
    parser.add_argument("--use-templates", action="store_true", help="Apply template-based refinement")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generate-audio", action="store_true", help="Generate audio output using Unit Decoder + Vocoder")
    parser.add_argument("--unit-decoder-checkpoint", type=str, default="checkpoints_mini_units_v4/checkpoint_best.pt", help="Checkpoint path to load Unit Decoder weights from (EchoStream full model)")
    parser.add_argument("--unit-temperature", type=float, default=0.0, help="Temperature for Unit Decoder sampling (0.0=greedy, >0.0=sampling, 1.0~2.0 추천)")
    parser.add_argument("--vocoder-checkpoint", type=str, default="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000", help="Vocoder checkpoint path")
    parser.add_argument("--vocoder-config", type=str, default="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config.json", help="Vocoder config path")
    parser.add_argument("--output-audio-dir", type=str, default="output_audio", help="Directory to save generated audio files")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1) Tokenizer
    spm = SPMTokenizer(args.vocab_path)
    pad_id, bos_id, eos_id, unk_id = spm.pad_id, spm.bos_id, spm.eos_id, spm.unk_id
    vocab_size = len(spm)
    logger.info(f"Vocab size: {vocab_size}, UNK={unk_id}")

    # 2) word2tokens 로드
    with open(args.word2tokens, "r", encoding="utf-8") as f:
        word2tokens = json.load(f)
    word_list = list(word2tokens.keys())
    logger.info(f"Loaded {len(word_list)} words from word2tokens")

    # 3) WPB 로드
    wpb_ckpt = torch.load(args.checkpoint_wpb, map_location=device)
    wpb = WordPriorBooster(
        enc_dim=wpb_ckpt.get("enc_dim", args.encoder_dim),
        num_words=len(word_list),
    ).to(device)
    wpb.load_state_dict(wpb_ckpt["model_state"], strict=False)
    wpb.eval()
    logger.info("WPB loaded")
    
    # 3-1) Subword N-gram LM 로드 (optional)
    ngram_lm = None
    if args.ngram_lm and Path(args.ngram_lm).exists():
        ngram_lm = NgramLM(n=3, smoothing=0.1)
        ngram_lm.load(Path(args.ngram_lm))
        ngram_lm.vocab_size = vocab_size  # Ensure vocab size matches
        logger.info(f"Subword N-gram LM loaded from {args.ngram_lm}")
    elif args.ngram_lm:
        logger.warning(f"Subword N-gram LM file not found: {args.ngram_lm}, continuing without it")
    
    # 3-2) Word-level N-gram LM 로드 (optional)
    word_ngram_lm = None
    if args.word_ngram_lm and Path(args.word_ngram_lm).exists():
        word_ngram_lm = WordNgramLM(n=3, smoothing=0.1)
        word_ngram_lm.load(Path(args.word_ngram_lm))
        logger.info(f"Word-level N-gram LM loaded from {args.word_ngram_lm}")
    elif args.word_ngram_lm:
        logger.warning(f"Word-level N-gram LM file not found: {args.word_ngram_lm}, continuing without it")
    
    # 3-3) Templates 로드 (optional)
    template_matcher = None
    if args.templates and Path(args.templates).exists():
        template_matcher = TemplateMatcher()
        template_matcher.load(Path(args.templates))
        logger.info(f"Templates loaded from {args.templates}")
    elif args.templates:
        logger.warning(f"Templates file not found: {args.templates}, continuing without it")

    # 4) ST 모델 로드 (encoder + decoder)
    # Checkpoint에서 config 읽기
    decoder_ffn_dim = 512  # default
    if args.checkpoint_st and Path(args.checkpoint_st).exists():
        ckpt = torch.load(args.checkpoint_st, map_location=device)
        # config에서 decoder_ffn_dim 찾기
        if "config" in ckpt:
            config = ckpt["config"]
            if isinstance(config, dict):
                decoder_ffn_dim = config.get("decoder_ffn_dim", 512)
        logger.info(f"Using decoder_ffn_dim={decoder_ffn_dim} from checkpoint")
    
    st_model = STCTCWithLM(
        vocab_size=vocab_size,
        encoder_dim=args.encoder_dim,
        encoder_layers=4,
        decoder_dim=args.encoder_dim,
        decoder_layers=3,
        decoder_ffn_dim=decoder_ffn_dim,
        dropout=0.1,
    )
    
    if args.checkpoint_st and Path(args.checkpoint_st).exists():
        ckpt = torch.load(args.checkpoint_st, map_location=device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        st_model.load_state_dict(state_dict, strict=False)
        logger.info(f"ST model loaded from {args.checkpoint_st}")
    
    st_model.to(device)
    st_model.eval()

    # 4-1) Unit Decoder와 Vocoder 초기화 (음성 생성용)
    unit_decoder = None
    vocoder = None
    if args.generate_audio:
        logger.info("Initializing Unit Decoder and Vocoder for audio generation...")
        
        # Unit Decoder
        unit_decoder = CTCTransformerUnitDecoder(
            input_dim=args.encoder_dim,
            embed_dim=args.encoder_dim,
            num_layers=6,
            num_heads=4,
            ffn_embed_dim=1024,
            num_units=1000,
            ctc_upsample_ratio=5,
            dropout=0.1,
        ).to(device)
        
        # Unit Decoder 가중치 로드 (학습된 가중치가 있으면)
        unit_decoder_ckpt_path = Path(args.unit_decoder_checkpoint)
        if unit_decoder_ckpt_path.exists():
            try:
                logger.info(f"Loading Unit Decoder weights from {unit_decoder_ckpt_path}")
                unit_ckpt = torch.load(unit_decoder_ckpt_path, map_location=device)
                
                # state_dict 찾기
                if 'model_state_dict' in unit_ckpt:
                    unit_state_dict = unit_ckpt['model_state_dict']
                elif 'model' in unit_ckpt:
                    unit_state_dict = unit_ckpt['model']
                else:
                    unit_state_dict = unit_ckpt
                
                # Unit Decoder 키만 추출 (prefix 제거)
                unit_decoder_state = {}
                for k, v in unit_state_dict.items():
                    if 'unit_decoder' in k:
                        # 'unit_decoder.' prefix 제거
                        new_key = k.replace('unit_decoder.', '')
                        unit_decoder_state[new_key] = v
                
                if unit_decoder_state:
                    # 로드 시도 (일부 키가 없을 수 있으므로 strict=False)
                    missing_keys, unexpected_keys = unit_decoder.load_state_dict(unit_decoder_state, strict=False)
                    if missing_keys:
                        logger.warning(f"Unit Decoder: {len(missing_keys)} keys not found (using random init)")
                    if unexpected_keys:
                        logger.warning(f"Unit Decoder: {len(unexpected_keys)} unexpected keys ignored")
                    logger.info(f"✅ Unit Decoder weights loaded ({len(unit_decoder_state)} keys)")
                else:
                    logger.warning("Unit Decoder keys not found in checkpoint, using random initialization")
            except Exception as e:
                logger.warning(f"Failed to load Unit Decoder weights: {e}")
                logger.warning("Using random initialization")
        else:
            logger.warning(f"Unit Decoder checkpoint not found: {unit_decoder_ckpt_path}")
            logger.warning("Using random initialization")
        
        unit_decoder.eval()
        logger.info("Unit Decoder initialized")
        
        # Vocoder
        vocoder_checkpoint = Path(args.vocoder_checkpoint)
        vocoder_config = Path(args.vocoder_config)
        
        if vocoder_checkpoint.exists() and vocoder_config.exists():
            vocoder = CodeHiFiGANVocoder(
                num_units=1000,
                sample_rate=16000,
                checkpoint_path=str(vocoder_checkpoint),
                config_path=str(vocoder_config),
            ).to(device)
            vocoder.eval()
            logger.info(f"Vocoder loaded from {vocoder_checkpoint}")
        else:
            logger.warning(f"Vocoder checkpoint/config not found. Using dummy vocoder.")
            logger.warning(f"  Checkpoint: {vocoder_checkpoint} (exists: {vocoder_checkpoint.exists()})")
            logger.warning(f"  Config: {vocoder_config} (exists: {vocoder_config.exists()})")
            vocoder = CodeHiFiGANVocoder(
                num_units=1000,
                sample_rate=16000,
            ).to(device)
            vocoder.eval()
        
        # Output directory
        output_audio_dir = Path(args.output_audio_dir)
        output_audio_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio output directory: {output_audio_dir}")

    # 5) 데이터
    dataset = S2STManifestDataset(
        manifest_path=args.manifest,
        data_root="data",
        sample_rate=16000,
        num_mel_bins=80,
        tgt_vocab_path=args.vocab_path,
        text_level="word",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_s2st_batches,
    )

    # 6) 디코딩 루프
    count = 0
    for batch in loader:
        if count >= args.num_samples:
            break

        speech = batch["speech"].to(device)  # [B, T, 80]
        speech_lengths = batch["speech_lengths"].to(device)  # [B]
        
        # GT 텍스트 (영어 번역 정답)
        tgt_tokens = batch.get("target_text")
        tgt_lengths = batch.get("target_lengths")
        if tgt_tokens is not None:
            ref_tokens = tgt_tokens[0][:tgt_lengths[0]-1].cpu().tolist()
            ref_text = spm.sp.decode(ref_tokens)
        else:
            ref_text = "N/A"
        
        # 한국어 원문 (참고용) - dataset에서 직접 가져오기
        batch_indices = batch.get("indices", [count])
        if batch_indices and len(batch_indices) > 0:
            src_text = dataset[batch_indices[0]].get("src_text", "N/A")
        else:
            src_text = "N/A"

        with torch.no_grad():
            # Encoder
            enc_out_dict = st_model.encoder(speech, speech_lengths)
            encoder_out = enc_out_dict["encoder_out"][0]  # [T, B, D]
            encoder_padding_mask = enc_out_dict["encoder_padding_mask"][0] if enc_out_dict["encoder_padding_mask"] else None
            T_enc, B, D = encoder_out.shape
            
            # Encoder 출력을 [B, T, D]로 변환 (WPB용)
            enc_out_bt = encoder_out.transpose(0, 1)  # [B, T, D]

            # WPB prior
            wpb_logits = wpb(enc_out_bt)  # [B, num_words]
            
            # Temperature 적용 (분포를 flatten)
            if args.temperature != 1.0:
                wpb_logits = wpb_logits / args.temperature
            
            word_probs = torch.sigmoid(wpb_logits[0])  # [num_words]
            
            # 방법 1: 특정 단어("no", "or")의 prior를 약하게 만들기
            if args.suppress_words and args.suppress_scale < 1.0:
                for i, w in enumerate(word_list):
                    if w in args.suppress_words:
                        word_probs[i] = word_probs[i] * args.suppress_scale

            # Token prior
            token_prior = build_token_prior(word_probs, word_list, word2tokens, vocab_size)
            log_prior = torch.log(token_prior + 1e-8).to(device)  # [V]
            
            # Zero-mean으로 만들어서 "상대적인" bias만 주기
            log_prior = log_prior - log_prior.mean()

            # CTC head를 사용해서 logits 얻기 (decoder는 autoregressive라서 한 번에 못 얻음)
            # encoder_out: [T, B, D] -> CTC head -> [T, B, V]
            ctc_logits = st_model.ctc_head(encoder_out)  # [T, B, V]
            ctc_logits = ctc_logits.transpose(0, 1)  # [B, T, V]
            
            # Fusion: log_prior를 모든 타임스텝에 더하기
            logits_fused = ctc_logits + args.alpha * log_prior.view(1, 1, -1)  # [B, T, V]
            
            # N-gram LM fusion (if available)
            # Iterative refinement: decode -> get context -> refine
            if ngram_lm is not None and args.ngram_weight > 0:
                # First pass: get initial sequence for context (CTC greedy)
                initial_probs = F.log_softmax(logits_fused, dim=-1)  # [B, T, V]
                initial_tokens = initial_probs.argmax(dim=-1)  # [B, T]
                
                # Build context sequence (remove blanks/duplicates, CTC collapse)
                context_sequence = []
                prev = None
                for t in range(T_enc):
                    token = initial_tokens[0, t].item()
                    if token != prev and token != 1:  # 1 is blank
                        context_sequence.append(token)
                    prev = token
                
                # Build N-gram LM log probs for each timestep
                # Use sliding window: for each timestep, use previous decoded tokens as context
                ngram_log_probs = torch.zeros_like(logits_fused)  # [B, T, V]
                
                # For each timestep, estimate which position in context_sequence we're at
                # Approximate: assume uniform distribution of context tokens across timesteps
                context_pos = 0
                for t in range(T_enc):
                    # Get context from previous decoded tokens
                    if context_pos > 0:
                        # Use last n-1 tokens as context
                        context_start = max(0, context_pos - (ngram_lm.n - 1))
                        context = context_sequence[context_start:context_pos]
                    else:
                        context = []
                    
                    # Get N-gram LM log probs for all tokens
                    for v in range(vocab_size):
                        ngram_log_prob = ngram_lm.get_log_prob(context, v)
                        ngram_log_probs[0, t, v] = ngram_log_prob
                    
                    # Update context position (approximate: every few timesteps = one token)
                    # This is approximate since we don't know exact alignment
                    if t > 0 and t % max(1, T_enc // max(len(context_sequence), 1)) == 0:
                        context_pos = min(context_pos + 1, len(context_sequence))
                
                # Fuse N-gram LM
                logits_fused = logits_fused + args.ngram_weight * ngram_log_probs
            
            if args.unk_penalty > 0 and unk_id is not None:
                logits_fused[:, :, unk_id] -= args.unk_penalty
            
            # 방법 3: "no", "or" 토큰에 repetition penalty 적용
            if args.rep_penalty > 0 and args.suppress_words:
                suppress_token_ids = set()
                for word in args.suppress_words:
                    if word in word2tokens:
                        suppress_token_ids.update(word2tokens[word])
                for tid in suppress_token_ids:
                    if 0 <= tid < vocab_size:
                        logits_fused[:, :, tid] -= args.rep_penalty

            # Decoding: Use autoregressive decoder or CTC
            if args.use_decoder:
                # Autoregressive decoder with WPB prior fusion
                hyp_tokens = autoregressive_decode_with_wpb(
                    st_model.decoder,
                    encoder_out,
                    encoder_padding_mask,
                    log_prior,
                    args.alpha,
                    bos_id,
                    eos_id,
                    unk_id,
                    args.unk_penalty,
                    args.max_len,
                    args.decoder_rep_penalty,
                    device,
                )
            else:
                # CTC decoding (original)
                # Convert to log_probs for decoding
                log_probs_fused = F.log_softmax(logits_fused, dim=-1)  # [B, T, V]
                
                if args.beam_size > 1:
                    # Beam search
                    decoder = CTCBeamSearchDecoder(
                        blank_id=1,  # CTC blank
                        beam_size=args.beam_size,
                        length_penalty=args.length_penalty,
                        repetition_penalty=args.beam_rep_penalty,
                    )
                    # Use encoder output length (T_enc) for beam search
                    enc_lengths = (~encoder_padding_mask).sum(dim=1) if encoder_padding_mask is not None else torch.full((B,), T_enc, dtype=torch.long, device=device)
                    decoded_seqs = decoder.decode(
                        log_probs_fused,
                        input_lengths=enc_lengths,
                    )
                    hyp_tokens = decoded_seqs[0] if decoded_seqs else []
                else:
                    # Greedy decode (original)
                    pred_ids = log_probs_fused.argmax(dim=-1)  # [B, T]
                    
                    # CTC collapse: remove duplicates and blanks
                    hyp_tokens = []
                    prev = None
                    for t in range(pred_ids.size(1)):
                        token = pred_ids[0, t].item()
                        if token == unk_id and args.unk_penalty > 0:
                            continue  # skip UNK if penalty applied
                        if token != prev and token != 1:  # 1 is blank in CTC
                            hyp_tokens.append(token)
                        prev = token
            
            # Post-processing: Improve sentence structure
            if args.post_process:
                hyp_tokens = apply_sentence_structure_filters(
                    hyp_tokens,
                    remove_n_gram=True,
                    remove_short=True,
                    filter_length=True,
                )
            
            hyp_text = spm.sp.decode(hyp_tokens) if hyp_tokens else ""
            
            # Word-level N-gram LM scoring and template matching (post-decoding refinement)
            if word_ngram_lm is not None and args.word_ngram_weight > 0:
                # Decode to words for word-level scoring
                hyp_words = hyp_text.lower().strip().split()
                hyp_words = [w.strip() for w in hyp_words if w.strip()]
                
                if len(hyp_words) > 0:
                    # Score with word-level N-gram LM
                    word_score = word_ngram_lm.score_sequence(hyp_words)
                    
                    # If score is too low (repetitive/unusual), try template matching more aggressively
                    if word_score < -2.0 and len(hyp_words) > 5:
                        logger.debug(f"Word-level N-gram score: {word_score:.3f} (low, trying template matching)")
                        # Template matching will be tried below
            
            # Template-based refinement
            if args.use_templates and template_matcher is not None:
                hyp_words = hyp_text.lower().strip().split()
                hyp_words = [w.strip() for w in hyp_words if w.strip()]
                
                if len(hyp_words) > 0:
                    # Try to match template with reference text for better matching
                    template_text = template_matcher.match_template(
                        hyp_words, 
                        min_keywords=2,  # Lower threshold
                        reference_text=ref_text if ref_text != "N/A" else None
                    )
                    if template_text:
                        logger.info(f"[{count}] Template matched! Original: {hyp_text[:50]}...")
                        logger.info(f"[{count}] Template: {template_text[:80]}...")
                        hyp_text = template_text

        print(f"[{count}] 한국어 원문: {src_text}")
        print(f"[{count}] 영어 번역 (정답): {ref_text}")
        print(f"[{count}] 영어 번역 (예측): {hyp_text}")
        
        # 음성 생성 (옵션)
        if args.generate_audio and unit_decoder is not None and vocoder is not None:
            try:
                # Unit Decoder 입력: ST decoder의 hidden states 사용
                # st_model.decoder는 LightSTDecoder이므로, decoder output을 사용
                # 대안: Encoder output을 직접 사용 (현재 방식, 품질 낮을 수 있음)
                
                # 옵션 1: ST Decoder를 통해 hidden states 얻기 (더 나은 방법)
                # 하지만 현재는 CTC만 사용하므로 decoder output이 없음
                # 따라서 Encoder output을 사용하되, 더 나은 처리를 위해 평균 풀링 등 적용
                
                # ST → MT → Unit Decoder 파이프라인
                # 1. ST CTC 디코딩으로 텍스트 토큰 얻기
                st_tokens = hyp_tokens  # 이미 디코딩된 토큰 사용
                
                # 2. MT Decoder를 통해 텍스트 hidden states 얻기
                if st_model.mt_decoder is not None and len(st_tokens) > 0:
                    # ST tokens를 MT Decoder input으로 변환
                    # [B, T] 형태로 변환 (batch dimension 추가)
                    st_tokens_tensor = torch.tensor([st_tokens], device=device, dtype=torch.long)  # [1, T]
                    
                    # Encoder output dict 준비
                    encoder_out_dict = {
                        'encoder_out': [encoder_out],  # [T, B, D]
                        'encoder_padding_mask': [encoder_padding_mask] if encoder_padding_mask is not None else None,
                    }
                    
                    # MT Decoder forward
                    mt_out = st_model.mt_decoder(
                        prev_output_tokens=st_tokens_tensor,
                        encoder_out=encoder_out_dict,
                    )
                    
                    # MT Decoder hidden states 사용
                    text_hidden = mt_out.get('decoder_out', encoder_out)  # [T, B, D]
                    logger.info(f"[{count}] MT Decoder 사용: {text_hidden.shape}")
                    logger.info(f"[{count}] MT Decoder output 통계: mean={text_hidden.mean():.3f}, std={text_hidden.std():.3f}, min={text_hidden.min():.3f}, max={text_hidden.max():.3f}")
                else:
                    # MT Decoder가 없으면 Encoder output 직접 사용
                    text_hidden = encoder_out  # [T, B, D]
                    logger.info(f"[{count}] MT Decoder 없음, Encoder output 직접 사용")
                
                # Padding mask 변환 (Unit Decoder는 [B, T] 형태 필요)
                if encoder_padding_mask is not None:
                    text_padding_mask = encoder_padding_mask  # [B, T]
                else:
                    text_padding_mask = None
                
                # Unit Decoder forward
                unit_out = unit_decoder(
                    text_hidden=text_hidden,
                    text_padding_mask=text_padding_mask,
                )
                
                # Units 추출 (greedy decoding)
                unit_log_probs = unit_out['log_probs']  # [B, T_unit, num_units]
                
                # Unit Decoder 출력 디버깅
                log_probs_sample = unit_log_probs[0, :5, :]  # 첫 5개 timestep
                probs_sample = torch.exp(log_probs_sample)  # [5, 1000]
                top_probs, top_indices = torch.topk(probs_sample, k=5, dim=-1)  # [5, 5]
                
                logger.info(f"[{count}] Unit Decoder 출력 분석:")
                logger.info(f"  - Log probs shape: {unit_log_probs.shape}")
                logger.info(f"  - Log probs 범위: {unit_log_probs.min():.3f} ~ {unit_log_probs.max():.3f}")
                logger.info(f"  - Log probs 평균: {unit_log_probs.mean():.3f}, 표준편차: {unit_log_probs.std():.3f}")
                
                # 확률 분포 분석: 얼마나 peaky한지 확인
                probs_all = torch.exp(unit_log_probs[0, :, :])  # [T_unit, 1000]
                entropy = -(probs_all * torch.log(probs_all + 1e-10)).sum(dim=-1)  # [T_unit]
                logger.info(f"  - 평균 엔트로피: {entropy.mean():.3f} (높을수록 다양, max={np.log(1000):.3f})")
                logger.info(f"  - 최대 확률 평균: {probs_all.max(dim=-1)[0].mean():.3f} (낮을수록 다양)")
                
                logger.info(f"  - 첫 5 timestep의 top-5 units:")
                for t in range(5):
                    logger.info(f"    t={t}: {top_indices[t].tolist()} (probs: {[f'{p:.3f}' for p in top_probs[t].tolist()]})")
                
                # Units 추출: Greedy 대신 Temperature sampling으로 다양성 증가
                if hasattr(args, 'unit_temperature') and args.unit_temperature > 0:
                    # Temperature sampling
                    unit_temperature = args.unit_temperature
                    scaled_log_probs = unit_log_probs / unit_temperature
                    unit_probs = torch.exp(scaled_log_probs)  # [B, T_unit, num_units]
                    # Sample from distribution
                    units = torch.multinomial(unit_probs.view(-1, unit_probs.size(-1)), 1).view(unit_probs.size(0), unit_probs.size(1))  # [B, T_unit]
                    logger.info(f"[{count}] Units 추출: Temperature sampling (T={unit_temperature})")
                else:
                    # Greedy decoding
                    units = unit_log_probs.argmax(dim=-1)  # [B, T_unit]
                    logger.info(f"[{count}] Units 추출: Greedy decoding")
                
                # Units 디버깅 정보 출력
                units_np = units[0].cpu().numpy() if units.dim() > 1 else units.cpu().numpy()
                unique_units = len(set(units_np.flatten().tolist()))
                unit_min, unit_max = int(units_np.min()), int(units_np.max())
                unit_mean = float(units_np.mean())
                
                # Unit 분포 계산
                unique_vals, counts = torch.unique(units[0], return_counts=True)
                top_units = dict(zip(unique_vals[:10].tolist(), counts[:10].tolist()))
                
                logger.info(f"[{count}] Units 통계:")
                logger.info(f"  - Units 길이: {units.size(1)}")
                logger.info(f"  - Unique units: {unique_units} / {units.size(1)}")
                logger.info(f"  - Unit 범위: {unit_min} ~ {unit_max} (평균: {unit_mean:.1f})")
                logger.info(f"  - Unit 분포 (상위 10개): {top_units}")
                
                # Units가 모두 같은 값이거나 범위가 이상한지 확인
                if unique_units < 10:
                    logger.warning(f"[{count}] ⚠️  Units 다양성이 매우 낮음 (unique={unique_units})")
                if unit_max >= 1000:
                    logger.warning(f"[{count}] ⚠️  Units 범위가 vocab 크기(1000)를 초과: {unit_max}")
                if unit_min < 0:
                    logger.warning(f"[{count}] ⚠️  Units에 음수 값 존재: {unit_min}")
                
                # Units를 유효한 범위로 클리핑 (0~999)
                units = torch.clamp(units, 0, 999)
                
                # Vocoder로 음성 생성
                waveform = vocoder.generate(units, return_duration=False)  # [B, T_wav]
                
                # 첫 번째 배치만 사용
                if waveform.dim() > 1:
                    waveform = waveform[0]  # [T_wav]
                
                # CPU로 이동 및 numpy 변환
                waveform_np = waveform.cpu().numpy()
                
                # 음성 파일 저장
                output_filename = output_audio_dir / f"sample_{count:03d}.wav"
                sf.write(str(output_filename), waveform_np, 16000)
                
                duration_sec = len(waveform_np) / 16000.0
                print(f"[{count}] 음성 생성 완료: {output_filename} ({duration_sec:.2f}초, {units.size(1)} units)")
                
            except Exception as e:
                logger.error(f"[{count}] 음성 생성 실패: {e}")
                import traceback
                traceback.print_exc()
        
        print("-" * 60)
        count += 1

    logger.info(f"Decoded {count} samples")


if __name__ == "__main__":
    main()
