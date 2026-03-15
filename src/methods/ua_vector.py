"""
Uncertainty-Aware (UA) CoT Vector implementation.
Bayesian MAP estimation with structured prior and adaptive gating.

Supports: qwen (plain text), qwen3 (ChatML), llama (Llama-3 template).
"""

import torch
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import (
    PROMPT_TEMPLATES,
    apply_chat_template,
    apply_chat_template_nothink,
    needs_chat_template,
)


class UACoTVector(BaseCoTVectorMethod):
    """
    Uncertainty-Aware CoT Vector with Bayesian shrinkage.
    z_d = k_d * μ_d where k_d = τ² / (σ²_d + τ²)
    """
    
    def __init__(self, model_wrapper, tokenizer, layer_idx, dataset_type="gsm8k",
                 tau_squared=1.0, min_variance=1e-6):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        self.tau_squared = tau_squared
        self.min_variance = min_variance
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        self.mean_vector = None
        self.variance_vector = None
        self.shrinkage_coefficients = None
    
    def _build_prompts(self, sample):
        """Build CoT and non-CoT prompts with proper chat template."""
        model_name = self.model_wrapper.model_name
        
        if self.dataset_type == "mmlu_pro":
            cot_raw = self.prompt_template["cot"].format(
                question=sample.question, choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            non_cot_raw = self.prompt_template["non_cot"].format(
                question=sample.question, choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            cot_raw = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            non_cot_raw = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        if needs_chat_template(model_name):
            cot_prompt = apply_chat_template(
                cot_raw, self.tokenizer, model_name, self.dataset_type
            )
            non_cot_prompt = apply_chat_template_nothink(
                non_cot_raw, self.tokenizer, model_name, self.dataset_type
            )
        else:
            cot_prompt = cot_raw
            non_cot_prompt = non_cot_raw
        
        return cot_prompt, non_cot_prompt
    
    def extract_single(self, sample) -> torch.Tensor:
        """Extract activation difference for a single sample."""
        device = self.model_wrapper.device
        
        cot_prompt, non_cot_prompt = self._build_prompts(sample)
        
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)
        
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)
        
        diff = cot_answer_activation - non_cot_answer_activation
        self.model_wrapper.clear_hooks()
        return diff.squeeze(0)
    
    def extract(self, support_samples: List) -> torch.Tensor:
        """Extract Uncertainty-Aware CoT Vector with Bayesian shrinkage."""
        print(f"Extracting UA CoT Vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        print(f"  Prior variance τ² = {self.tau_squared}")
        
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception:
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        stacked = torch.stack(vectors, dim=0)
        self.mean_vector = stacked.mean(dim=0)
        
        if len(vectors) > 1:
            self.variance_vector = stacked.var(dim=0, unbiased=True)
        else:
            self.variance_vector = torch.full_like(self.mean_vector, self.min_variance)
        
        self.variance_vector = torch.clamp(self.variance_vector, min=self.min_variance)
        self.shrinkage_coefficients = self.tau_squared / (self.variance_vector + self.tau_squared)
        ua_vector = self.shrinkage_coefficients * self.mean_vector
        self.vector = ua_vector
        self._print_statistics()
        return ua_vector
    
    def _print_statistics(self):
        print(f"\nUA Vector Statistics:")
        print(f"  Mean vector norm: {self.mean_vector.norm().item():.4f}")
        print(f"  UA vector norm: {self.vector.norm().item():.4f}")
        print(f"  Shrinkage ratio: {(self.vector.norm() / self.mean_vector.norm()).item():.4f}")
        k = self.shrinkage_coefficients
        print(f"\nShrinkage Coefficients (k):")
        print(f"  Mean: {k.mean().item():.4f}, Std: {k.std().item():.4f}")
        print(f"  Min: {k.min().item():.4f}, Max: {k.max().item():.4f}")
        highly_suppressed = (k < 0.1).sum().item()
        moderately_kept = ((k >= 0.1) & (k < 0.5)).sum().item()
        well_preserved = (k >= 0.5).sum().item()
        total_dims = k.numel()
        print(f"\nDimension Classification:")
        print(f"  Highly suppressed (k < 0.1): {highly_suppressed} ({100*highly_suppressed/total_dims:.1f}%)")
        print(f"  Moderately kept (0.1 ≤ k < 0.5): {moderately_kept} ({100*moderately_kept/total_dims:.1f}%)")
        print(f"  Well preserved (k ≥ 0.5): {well_preserved} ({100*well_preserved/total_dims:.1f}%)")
    
    def get_vector(self):
        return self.vector
    
    def get_statistics(self):
        if self.mean_vector is None:
            return {}
        return {
            "mean_vector": self.mean_vector.cpu(),
            "variance_vector": self.variance_vector.cpu(),
            "shrinkage_coefficients": self.shrinkage_coefficients.cpu(),
            "tau_squared": self.tau_squared,
            "ua_vector": self.vector.cpu() if self.vector is not None else None,
        }
