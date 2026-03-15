"""
Extracted CoT Vector implementation.
Implements Eq. 4 and 5 from the paper.

Supports: qwen (plain text), qwen3 (ChatML), llama (Llama-3 template).
"""

import torch
from typing import List, Optional
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import (
    PROMPT_TEMPLATES,
    apply_chat_template,
    apply_chat_template_nothink,
    needs_chat_template,
)


class ExtractedCoTVector(BaseCoTVectorMethod):
    """
    Extract CoT Vector by computing activation differences.
    v_CoT = (1/N) * Σ (α_CoT(a) - α_NonCoT(a))
    """
    
    def __init__(self, model_wrapper, tokenizer, layer_idx, dataset_type="gsm8k"):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def _build_prompts(self, sample):
        """Build CoT and non-CoT prompts with proper chat template."""
        model_name = self.model_wrapper.model_name
        
        # Build raw prompts
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
        
        # Apply chat template if needed
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
        """Extract CoT vector from a single sample."""
        device = self.model_wrapper.device
        
        cot_prompt, non_cot_prompt = self._build_prompts(sample)
        
        # Tokenize
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        # Find answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        # Extract CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)
        
        # Extract non-CoT activations
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
        """Extract task-general CoT vector from support set (Eq. 5)."""
        print(f"Extracting CoT vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        
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
        task_vector = stacked.mean(dim=0)
        self.vector = task_vector
        print(f"Extracted vector: shape={task_vector.shape}, norm={task_vector.norm().item():.4f}")
        return task_vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        return self.vector
