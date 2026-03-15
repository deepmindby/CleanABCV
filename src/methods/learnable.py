"""
Learnable CoT Vector implementation.
Teacher-student framework from Section 3.2.2.

Supports: qwen (plain text), qwen3 (ChatML), llama (Llama-3 template).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math
import gc

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import (
    PROMPT_TEMPLATES,
    apply_chat_template,
    apply_chat_template_nothink,
    needs_chat_template,
)


class CoTDataset(Dataset):
    """Dataset for CoT vector training with model-specific templates."""
    
    def __init__(self, samples, tokenizer, dataset_type, model_name, max_length=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.model_name = model_name
        self.max_length = max_length
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build raw prompts
        if self.dataset_type == "mmlu_pro":
            teacher_raw = self.prompt_template["cot"].format(
                question=sample.question, choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            student_raw = self.prompt_template["non_cot"].format(
                question=sample.question, choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            teacher_raw = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            student_raw = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Apply chat template
        if needs_chat_template(self.model_name):
            teacher_prompt = apply_chat_template(
                teacher_raw, self.tokenizer, self.model_name, self.dataset_type
            )
            student_prompt = apply_chat_template_nothink(
                student_raw, self.tokenizer, self.model_name, self.dataset_type
            )
        else:
            teacher_prompt = teacher_raw
            student_prompt = student_raw
        
        # Tokenize
        teacher_enc = self.tokenizer(
            teacher_prompt, return_tensors="pt",
            truncation=True, max_length=self.max_length,
        )
        student_enc = self.tokenizer(
            student_prompt, return_tensors="pt",
            truncation=True, max_length=self.max_length,
        )
        
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        answer_len = len(answer_ids)
        
        teacher_len = teacher_enc["input_ids"].shape[1]
        student_len = student_enc["input_ids"].shape[1]
        
        return {
            "teacher_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_mask": teacher_enc["attention_mask"].squeeze(0),
            "student_ids": student_enc["input_ids"].squeeze(0),
            "student_mask": student_enc["attention_mask"].squeeze(0),
            "teacher_len": teacher_len,
            "student_len": student_len,
            "answer_len": answer_len,
        }


def collate_fn(batch):
    """Custom collate function with dynamic padding."""
    max_teacher_len = max(item["teacher_len"] for item in batch)
    max_student_len = max(item["student_len"] for item in batch)
    
    teacher_ids_list, teacher_mask_list = [], []
    student_ids_list, student_mask_list = [], []
    teacher_lens, student_lens, answer_lens = [], [], []
    
    for item in batch:
        t_ids, t_mask = item["teacher_ids"], item["teacher_mask"]
        t_pad = max_teacher_len - len(t_ids)
        if t_pad > 0:
            t_ids = F.pad(t_ids, (0, t_pad), value=0)
            t_mask = F.pad(t_mask, (0, t_pad), value=0)
        teacher_ids_list.append(t_ids)
        teacher_mask_list.append(t_mask)
        
        s_ids, s_mask = item["student_ids"], item["student_mask"]
        s_pad = max_student_len - len(s_ids)
        if s_pad > 0:
            s_ids = F.pad(s_ids, (0, s_pad), value=0)
            s_mask = F.pad(s_mask, (0, s_pad), value=0)
        student_ids_list.append(s_ids)
        student_mask_list.append(s_mask)
        
        teacher_lens.append(item["teacher_len"])
        student_lens.append(item["student_len"])
        answer_lens.append(item["answer_len"])
    
    return {
        "teacher_ids": torch.stack(teacher_ids_list),
        "teacher_mask": torch.stack(teacher_mask_list),
        "student_ids": torch.stack(student_ids_list),
        "student_mask": torch.stack(student_mask_list),
        "teacher_len": teacher_lens,
        "student_len": student_lens,
        "answer_len": answer_lens,
    }


class LearnableCoTVector(BaseCoTVectorMethod):
    """Learnable CoT Vector optimized via teacher-student framework."""
    
    def __init__(self, model_wrapper, tokenizer, layer_idx, dataset_type="gsm8k",
                 lambda_val=0.5, learning_rate=5e-3, weight_decay=1e-3,
                 warmup_ratio=0.5, num_epochs=5, batch_size=2,
                 gradient_accumulation_steps=2, max_length=1024):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.lambda_val = lambda_val
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        self.learning_rate = self._get_tiered_learning_rate(
            model_wrapper.model_name, layer_idx, learning_rate
        )
        
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.vector_param, std=0.02)
    
    def _get_tiered_learning_rate(self, model_name, layer_idx, default_lr):
        if model_name.lower() in ("qwen", "qwen3"):
            lr = 5e-3 if layer_idx < 4 else 1e-4
            print(f"  Tiered LR (Qwen, layer {layer_idx}): {lr}")
            return lr
        elif model_name.lower() == "llama":
            lr = 1e-4
            print(f"  Tiered LR (LLaMA, layer {layer_idx}): {lr}")
            return lr
        else:
            print(f"  Using default LR for '{model_name}': {default_lr}")
            return default_lr
    
    def _compute_alignment_loss(self, teacher_hidden, student_hidden):
        teacher_probs = F.softmax(teacher_hidden.detach(), dim=-1)
        student_log_probs = F.log_softmax(student_hidden, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    def _compute_ce_loss(self, logits, labels, mask):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).float()
        
        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        return (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
    
    def train(self, support_samples, wandb_run=None):
        """Train the learnable CoT vector."""
        print(f"Training learnable vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  LR: {self.learning_rate}, λ: {self.lambda_val}")
        
        # Create dataset with model-specific templates
        dataset = CoTDataset(
            support_samples, self.tokenizer, self.dataset_type,
            self.model_wrapper.model_name, max_length=self.max_length,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, collate_fn=collate_fn,
        )
        
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        self.vector_param.data = self.vector_param.data.to(target_device)
        
        optimizer = torch.optim.AdamW(
            [self.vector_param], lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_ce = 0.0
            num_batches = 0
            optimizer.zero_grad()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    teacher_ids = batch["teacher_ids"].to(target_device)
                    teacher_mask = batch["teacher_mask"].to(target_device)
                    student_ids = batch["student_ids"].to(target_device)
                    student_mask = batch["student_mask"].to(target_device)
                    bs = teacher_ids.size(0)
                    
                    # Teacher forward
                    self.model_wrapper.clear_hooks()
                    self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
                    with torch.no_grad():
                        self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
                    teacher_hidden_raw = self.model_wrapper.get_activations(self.layer_idx)
                    
                    teacher_answer_hiddens = []
                    for i in range(bs):
                        t_len = batch["teacher_len"][i]
                        a_len = batch["answer_len"][i]
                        t_ans_pos = list(range(max(0, t_len - a_len), t_len))
                        if t_ans_pos:
                            teacher_answer_hiddens.append(
                                teacher_hidden_raw[i, t_ans_pos, :].mean(dim=0)
                            )
                    
                    self.model_wrapper.clear_hooks()
                    del teacher_hidden_raw
                    if not teacher_answer_hiddens:
                        continue
                    teacher_hidden = torch.stack(teacher_answer_hiddens)
                    del teacher_answer_hiddens
                    
                    # Student forward
                    self.model_wrapper.register_injection_hook(
                        self.layer_idx, self.vector_param, 1.0, requires_grad=True
                    )
                    self.model_wrapper.register_extraction_hook(
                        self.layer_idx, requires_grad=True
                    )
                    student_outputs = self.model_wrapper(student_ids, attention_mask=student_mask)
                    student_hidden_raw = self.model_wrapper.get_activations(self.layer_idx)
                    student_logits = student_outputs.logits
                    
                    student_answer_hiddens = []
                    ce_losses = []
                    
                    for i in range(bs):
                        s_len = batch["student_len"][i]
                        a_len = batch["answer_len"][i]
                        s_ans_pos = list(range(max(0, s_len - a_len), s_len))
                        if s_ans_pos and i < len(teacher_hidden):
                            student_answer_hiddens.append(
                                student_hidden_raw[i, s_ans_pos, :].mean(dim=0)
                            )
                            ans_mask = torch.zeros(student_mask.shape[1], device=target_device)
                            ans_mask[s_ans_pos] = 1
                            ce_losses.append(self._compute_ce_loss(
                                student_logits[i:i+1], student_ids[i:i+1], ans_mask.unsqueeze(0)
                            ))
                    
                    if not student_answer_hiddens:
                        self.model_wrapper.clear_hooks()
                        continue
                    
                    student_hidden = torch.stack(student_answer_hiddens)
                    teacher_hidden_filtered = teacher_hidden[:len(student_hidden)]
                    
                    align_loss = self._compute_alignment_loss(teacher_hidden_filtered, student_hidden)
                    ce_loss = torch.stack(ce_losses).mean() if ce_losses else torch.tensor(0.0, device=target_device)
                    
                    loss = (align_loss + self.lambda_val * ce_loss) / self.gradient_accumulation_steps
                    loss.backward()
                    
                    self.model_wrapper.clear_hooks()
                    del student_hidden_raw, student_logits, student_outputs
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_([self.vector_param], 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        if global_step % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    epoch_align += align_loss.item()
                    epoch_ce += ce_loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{epoch_loss/num_batches:.4f}",
                                      "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n  Warning: OOM at batch {batch_idx}")
                        self.model_wrapper.clear_hooks()
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    raise
            
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_([self.vector_param], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_align = epoch_align / max(num_batches, 1)
            avg_ce = epoch_ce / max(num_batches, 1)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, align={avg_align:.4f}, ce={avg_ce:.4f}")
            
            if wandb_run:
                wandb_run.log({"epoch": epoch+1, "train/loss": avg_loss,
                               "train/align_loss": avg_align, "train/ce_loss": avg_ce,
                               "train/lr": scheduler.get_last_lr()[0]})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.vector = self.vector_param.detach().clone()
            
            torch.cuda.empty_cache()
            gc.collect()
        
        if self.vector is None:
            self.vector = self.vector_param.detach().clone()
        print(f"Training complete. Vector norm: {self.vector.norm().item():.4f}")
        return self.vector
    
    def get_vector(self):
        return self.vector
