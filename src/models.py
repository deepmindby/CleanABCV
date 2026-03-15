"""
Model wrapper with hook mechanisms for CoT Vector injection and extraction.

Supports: qwen (Qwen2.5-Math), qwen3 (Qwen3-8B), llama (Llama-3-Instruct)
All three use the same .model.model.layers structure.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Callable


# All supported model families use the same layer access pattern
SUPPORTED_MODELS = {"qwen", "qwen3", "llama"}


class CoTModelWrapper(nn.Module):
    """
    Wrapper around HuggingFace models that provides:
    1. Forward hooks for extracting activations
    2. Injection hooks for adding CoT vectors
    """
    
    def __init__(self, model_path: str, model_name: str = "qwen"):
        super().__init__()
        
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Supported: {SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.model_path = model_path
        
        # Validate path
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found: {model_path}\n"
                f"Please check that the path exists and contains model files."
            )
        
        # Load model with multi-GPU support
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        
        # Hook management
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, torch.Tensor] = {}
        self._injection_vector_cached: Optional[torch.Tensor] = None
        
    def _get_num_layers(self) -> int:
        # All supported models: qwen, qwen3, llama use .model.model.layers
        return len(self.model.model.layers)
    
    def _get_hidden_size(self) -> int:
        return self.model.config.hidden_size
    
    def _get_layer(self, layer_idx: int) -> nn.Module:
        # All supported models use the same access pattern
        return self.model.model.layers[layer_idx]
    
    def register_extraction_hook(
        self, 
        layer_idx: int, 
        position_ids: Optional[torch.Tensor] = None,
        requires_grad: bool = False
    ):
        """Register hook to extract activations at specified layer."""
        layer = self._get_layer(layer_idx)
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            if position_ids is not None:
                extracted = hidden_states[:, position_ids, :]
            else:
                extracted = hidden_states
            
            if requires_grad:
                self._activations[layer_idx] = extracted.float()
            else:
                self._activations[layer_idx] = extracted.detach().float()
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def register_injection_hook(
        self, 
        layer_idx: int, 
        vector: torch.Tensor, 
        scaling_factor: float = 1.0,
        requires_grad: bool = False
    ):
        """Register hook to inject CoT vector at specified layer."""
        layer = self._get_layer(layer_idx)
        
        target_device = next(layer.parameters()).device
        target_dtype = next(layer.parameters()).dtype
        
        if requires_grad:
            self._injection_vector_raw = vector
            self._injection_scaling_factor = scaling_factor
        else:
            vector_scaled = scaling_factor * vector.to(device=target_device, dtype=target_dtype)
            if vector_scaled.dim() == 1:
                vector_scaled = vector_scaled.unsqueeze(0).unsqueeze(0)
            elif vector_scaled.dim() == 2:
                vector_scaled = vector_scaled.unsqueeze(0)
            self._injection_vector_cached = vector_scaled
        
        self._injection_requires_grad = requires_grad
        self._injection_target_device = target_device
        self._injection_target_dtype = target_dtype
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            if self._injection_requires_grad:
                vec = self._injection_scaling_factor * self._injection_vector_raw
                vec = vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0).unsqueeze(0)
                elif vec.dim() == 2:
                    vec = vec.unsqueeze(0)
                modified = hidden_states + vec.expand_as(hidden_states)
            else:
                modified = hidden_states + self._injection_vector_cached.expand_as(hidden_states)
            
            if rest is not None:
                return (modified,) + rest
            return modified
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._activations.get(layer_idx)
    
    def clear_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._activations.clear()
        self._injection_vector_cached = None
        if hasattr(self, '_injection_vector_raw'):
            self._injection_vector_raw = None
        if hasattr(self, '_injection_scaling_factor'):
            self._injection_scaling_factor = None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property 
    def dtype(self):
        return next(self.model.parameters()).dtype


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer with proper configuration.
    
    Includes fallback: if fast tokenizer fails (e.g. vocab/merges file issues),
    retries with use_fast=False (pure-Python tokenizer).
    """
    import os
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please check that the path exists and contains model files."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        # Common failure: fast tokenizer can't load vocab/merges from certain model saves
        import logging
        logging.getLogger("cot_vectors").warning(
            f"Fast tokenizer failed ({e}), retrying with use_fast=False"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
