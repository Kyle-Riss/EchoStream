"""
Language Model Wrapper for EchoStream

StreamSpeech 방식의 LM 통합을 위한 래퍼 클래스.
Fairseq LM 모델과 호환되는 인터페이스 제공.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class LanguageModelWrapper(nn.Module):
    """
    Language Model Wrapper for EchoStream.
    
    StreamSpeech의 sequence_generator.py와 호환되는 인터페이스 제공.
    Fairseq LM 모델을 래핑하여 사용.
    """
    
    def __init__(
        self,
        lm_model: nn.Module,
        tgt_dict=None,  # Target dictionary (for token mapping if needed)
    ):
        super().__init__()
        self.lm_model = lm_model
        self.tgt_dict = tgt_dict
        
        # Set to eval mode
        self.lm_model.eval()
        
        logger.info("LanguageModelWrapper initialized")
    
    def forward(
        self,
        tokens: torch.Tensor,  # [B, T] previous tokens
        incremental_state: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through language model.
        
        Args:
            tokens: [B, T] input tokens
            incremental_state: Optional incremental state for caching
        
        Returns:
            Dict with 'logits': [B, T, vocab_size]
        """
        # LM forward (Fairseq LM interface)
        # Fairseq LM models typically return logits directly or in a dict
        with torch.no_grad():
            if hasattr(self.lm_model, 'forward'):
                # Try standard forward
                lm_output = self.lm_model(tokens, incremental_state=incremental_state)
            else:
                raise ValueError("LM model does not have forward method")
            
            # Handle different output formats
            if isinstance(lm_output, dict):
                if 'logits' in lm_output:
                    logits = lm_output['logits']
                elif 'decoder_out' in lm_output:
                    # Some models return decoder_out
                    decoder_out = lm_output['decoder_out']
                    if isinstance(decoder_out, list):
                        decoder_out = decoder_out[0]
                    # Extract logits from decoder output
                    if hasattr(self.lm_model, 'output_projection'):
                        logits = self.lm_model.output_projection(decoder_out)
                    else:
                        # Assume decoder_out is already logits
                        logits = decoder_out
                else:
                    raise ValueError(f"Unknown LM output format: {lm_output.keys()}")
            else:
                # Direct logits
                logits = lm_output
            
            # Ensure correct shape: [B, T, vocab_size]
            if logits.dim() == 2:
                # [T, vocab] -> [1, T, vocab] (batch dimension)
                logits = logits.unsqueeze(0)
            elif logits.dim() == 3 and logits.size(0) != tokens.size(0):
                # [T, B, vocab] -> [B, T, vocab]
                if logits.size(1) == tokens.size(0):
                    logits = logits.transpose(0, 1)
            
            return {'logits': logits}
    
    def get_normalized_probs(
        self,
        net_output: Dict[str, torch.Tensor],
        log_probs: bool = True,
        sample: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Get normalized probabilities from LM output.
        
        StreamSpeech 방식: sequence_generator.py Line 350-352
        
        Args:
            net_output: Dict with 'logits' key
            log_probs: If True, return log probabilities
            sample: Optional sample dict (for compatibility)
        
        Returns:
            Normalized probabilities: [B, T, vocab_size]
        """
        logits = net_output['logits']  # [B, T, vocab_size]
        
        if log_probs:
            return torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            return torch.nn.functional.softmax(logits, dim=-1)


def load_language_model(
    lm_path: str,
    tgt_dict=None,
    device: str = "cuda",
) -> Optional[LanguageModelWrapper]:
    """
    Load language model from checkpoint.
    
    Args:
        lm_path: Path to LM checkpoint
        tgt_dict: Target dictionary (optional)
        device: Device to load model on
    
    Returns:
        LanguageModelWrapper or None if loading fails
    """
    try:
        # Try to load Fairseq LM
        import sys
        import os
        # Add fairseq to path if needed
        fairseq_path = os.path.join(
            os.path.dirname(__file__), '..', 'StreamSpeech_analysis', 'fairseq'
        )
        if os.path.exists(fairseq_path):
            sys.path.insert(0, fairseq_path)
        
        from fairseq import checkpoint_utils, utils
        from fairseq import tasks
        
        # Load checkpoint
        state = checkpoint_utils.load_checkpoint_to_cpu(lm_path)
        
        # Setup task
        task_args = state["cfg"]["task"]
        task = tasks.setup_task(task_args)
        
        # Load model
        models, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(lm_path),
            task=task,
        )
        
        lm_model = models[0]
        lm_model.eval()
        lm_model = lm_model.to(device)
        
        wrapper = LanguageModelWrapper(lm_model, tgt_dict=tgt_dict)
        logger.info(f"Language Model loaded from {lm_path}")
        
        return wrapper
        
    except Exception as e:
        logger.warning(f"Failed to load language model from {lm_path}: {e}")
        logger.warning("Continuing without language model")
        return None

