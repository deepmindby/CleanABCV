"""
Adaptive Bayesian CoT Vector (ABC Vector) implementation.

Prior Network p_phi(z|Q): predicts z distribution from question only
Posterior Network q_psi(z|Q,Y): uses privileged teacher features (train-only)
Gated injection: H_tilde = H + g * z

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
    get_terminators,
    needs_chat_template,
)
from ..utils import extract_answer_from_text, compare_answers


# ==================== MLP Networks ====================

class PriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q):
        h = self.net(r_Q)
        return self.mu_head(h), self.sigma_head(h)


class PosteriorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.mu_head.weight, std=0.01)
        nn.init.zeros_(self.sigma_head.bias)
        nn.init.normal_(self.sigma_head.weight, std=0.01)
    
    def forward(self, r_Q, Y):
        x = torch.cat([r_Q, Y], dim=-1)
        h = self.net(x)
        return self.mu_head(h), self.sigma_head(h)


# ==================== Dataset ====================

class ABCDataset(Dataset):
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
            question_raw = self.prompt_template["non_cot"].format(
                question=sample.question, choices=sample.choices
            )
        else:
            teacher_raw = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            student_raw = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
            question_raw = self.prompt_template["non_cot"].format(
                question=sample.question
            )
        
        # Apply chat templates
        if needs_chat_template(self.model_name):
            teacher_prompt = apply_chat_template(
                teacher_raw, self.tokenizer, self.model_name, self.dataset_type
            )
            student_prompt = apply_chat_template_nothink(
                student_raw, self.tokenizer, self.model_name, self.dataset_type
            )
            question_prompt = apply_chat_template_nothink(
                question_raw, self.tokenizer, self.model_name, self.dataset_type
            )
        else:
            teacher_prompt = teacher_raw
            student_prompt = student_raw
            question_prompt = question_raw
        
        # Tokenize
        teacher_enc = self.tokenizer(
            teacher_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        student_enc = self.tokenizer(
            student_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        question_enc = self.tokenizer(
            question_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        return {
            "teacher_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_mask": teacher_enc["attention_mask"].squeeze(0),
            "student_ids": student_enc["input_ids"].squeeze(0),
            "student_mask": student_enc["attention_mask"].squeeze(0),
            "question_ids": question_enc["input_ids"].squeeze(0),
            "question_mask": question_enc["attention_mask"].squeeze(0),
            "teacher_len": teacher_enc["input_ids"].shape[1],
            "student_len": student_enc["input_ids"].shape[1],
            "question_len": question_enc["input_ids"].shape[1],
            "answer_len": len(answer_ids),
        }


def abc_collate_fn(batch):
    max_t = max(item["teacher_len"] for item in batch)
    max_s = max(item["student_len"] for item in batch)
    max_q = max(item["question_len"] for item in batch)
    
    result = {k: [] for k in ["teacher_ids", "teacher_mask", "student_ids", "student_mask",
                               "question_ids", "question_mask"]}
    lens = {"teacher_len": [], "student_len": [], "question_len": [], "answer_len": []}
    
    for item in batch:
        for prefix, max_len in [("teacher", max_t), ("student", max_s), ("question", max_q)]:
            ids = item[f"{prefix}_ids"]
            mask = item[f"{prefix}_mask"]
            pad = max_len - len(ids)
            if pad > 0:
                ids = F.pad(ids, (0, pad), value=0)
                mask = F.pad(mask, (0, pad), value=0)
            result[f"{prefix}_ids"].append(ids)
            result[f"{prefix}_mask"].append(mask)
        for k in lens:
            lens[k].append(item[k])
    
    for k in result:
        result[k] = torch.stack(result[k])
    result.update(lens)
    return result


# ==================== Utility ====================

def compute_kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    kl = 0.5 * (torch.log(var_p / var_q) + var_q / var_p + ((mu_q - mu_p)**2) / var_p - 1.0)
    return kl.sum(dim=-1)


# ==================== ABC Vector ====================

class ABCCoTVector(BaseCoTVectorMethod):
    """Adaptive Bayesian CoT Vector with variational inference."""
    
    def __init__(self, model_wrapper, tokenizer, layer_idx, dataset_type="gsm8k",
                 abc_hidden_dim=512, kl_beta=1.0, kl_warmup_steps=0, sigma_min=1e-4,
                 learning_rate=1e-4, weight_decay=1e-3, warmup_ratio=0.1,
                 num_epochs=5, batch_size=2, gradient_accumulation_steps=2, max_length=1024):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.abc_hidden_dim = abc_hidden_dim
        self.kl_beta = kl_beta
        self.kl_warmup_steps = kl_warmup_steps
        self.sigma_min = sigma_min
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        hidden_size = model_wrapper.hidden_size
        self.hidden_size = hidden_size
        self.z_dim = hidden_size
        
        self.prior_net = PriorNetwork(hidden_size, abc_hidden_dim, self.z_dim)
        self.posterior_net = PosteriorNetwork(2 * hidden_size, abc_hidden_dim, self.z_dim)
        self.gate = nn.Parameter(torch.tensor(0.05))
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        self.trained = False
    
    def _get_sigma(self, raw_sigma):
        return F.softplus(raw_sigma) + self.sigma_min
    
    def _extract_question_repr(self, question_ids, question_mask):
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        with torch.no_grad():
            self.model_wrapper(question_ids, attention_mask=question_mask)
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        mask_expanded = question_mask.unsqueeze(-1).float()
        r_Q = (hidden_states * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        self.model_wrapper.clear_hooks()
        return r_Q.detach()
    
    def _extract_teacher_features(self, teacher_ids, teacher_mask, teacher_lens, answer_lens):
        bs = teacher_ids.size(0)
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx, requires_grad=False)
        with torch.no_grad():
            self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
        hidden_states = self.model_wrapper.get_activations(self.layer_idx)
        
        Y_list = []
        for i in range(bs):
            t_len = teacher_lens[i]
            a_len = answer_lens[i]
            ans_start = max(0, t_len - a_len)
            if ans_start < t_len:
                Y_list.append(hidden_states[i, ans_start:t_len, :].mean(dim=0))
            else:
                Y_list.append(hidden_states[i, t_len - 1, :])
        
        self.model_wrapper.clear_hooks()
        return torch.stack(Y_list, dim=0).detach()
    
    def _compute_ce_loss(self, logits, labels, mask):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        ce = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='none')
        flat_mask = shift_mask.view(-1).float()
        return (ce * flat_mask).sum() / (flat_mask.sum() + 1e-8)
    
    def _move_networks_to_device(self, device):
        self.prior_net = self.prior_net.to(device)
        self.posterior_net = self.posterior_net.to(device)
        self.gate.data = self.gate.data.to(device)
    
    def train(self, support_samples, wandb_run=None):
        print(f"Training ABC Vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  ABC Config: hidden_dim={self.abc_hidden_dim}, kl_beta={self.kl_beta}, "
              f"kl_warmup={self.kl_warmup_steps}, sigma_min={self.sigma_min}")
        
        dataset = ABCDataset(
            support_samples, self.tokenizer, self.dataset_type,
            self.model_wrapper.model_name, max_length=self.max_length,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=False, collate_fn=abc_collate_fn,
        )
        
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        self._move_networks_to_device(target_device)
        
        params = list(self.prior_net.parameters()) + list(self.posterior_net.parameters()) + [self.gate]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
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
            epoch_loss = epoch_nll = epoch_kl = 0.0
            num_batches = 0
            optimizer.zero_grad()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    teacher_ids = batch["teacher_ids"].to(target_device)
                    teacher_mask = batch["teacher_mask"].to(target_device)
                    student_ids = batch["student_ids"].to(target_device)
                    student_mask = batch["student_mask"].to(target_device)
                    question_ids = batch["question_ids"].to(target_device)
                    question_mask = batch["question_mask"].to(target_device)
                    bs = teacher_ids.size(0)
                    
                    r_Q = self._extract_question_repr(question_ids, question_mask)
                    Y = self._extract_teacher_features(
                        teacher_ids, teacher_mask, batch["teacher_len"], batch["answer_len"])
                    
                    mu_phi, raw_sigma_phi = self.prior_net(r_Q)
                    sigma_phi = self._get_sigma(raw_sigma_phi)
                    mu_psi, raw_sigma_psi = self.posterior_net(r_Q, Y)
                    sigma_psi = self._get_sigma(raw_sigma_psi)
                    
                    eps = torch.randn_like(mu_psi)
                    z = mu_psi + eps * sigma_psi
                    gated_z = self.gate * z
                    
                    nll_losses = []
                    for i in range(bs):
                        self.model_wrapper.clear_hooks()
                        self.model_wrapper.register_injection_hook(
                            self.layer_idx, vector=gated_z[i], scaling_factor=1.0, requires_grad=True)
                        outputs = self.model_wrapper(student_ids[i:i+1], attention_mask=student_mask[i:i+1])
                        
                        s_len = batch["student_len"][i]
                        a_len = batch["answer_len"][i]
                        ans_mask = torch.zeros(student_ids.shape[1], device=target_device)
                        ans_mask[max(0, s_len - a_len):s_len] = 1.0
                        
                        nll_losses.append(self._compute_ce_loss(
                            outputs.logits, student_ids[i:i+1], ans_mask.unsqueeze(0)))
                        self.model_wrapper.clear_hooks()
                    
                    nll_loss = torch.stack(nll_losses).mean()
                    kl_loss = compute_kl_divergence(mu_psi, sigma_psi, mu_phi, sigma_phi).mean()
                    
                    beta_t = self.kl_beta * min(1.0, global_step / self.kl_warmup_steps) \
                             if self.kl_warmup_steps > 0 else self.kl_beta
                    
                    loss = (nll_loss + beta_t * kl_loss) / self.gradient_accumulation_steps
                    loss.backward()
                    del r_Q, Y, z, gated_z, nll_losses
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        if global_step % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    epoch_nll += nll_loss.item()
                    epoch_kl += kl_loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{epoch_loss/num_batches:.4f}",
                                      "nll": f"{epoch_nll/num_batches:.4f}",
                                      "kl": f"{epoch_kl/num_batches:.4f}",
                                      "g": f"{self.gate.item():.3f}"})
                    
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
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_nll = epoch_nll / max(num_batches, 1)
            avg_kl = epoch_kl / max(num_batches, 1)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, nll={avg_nll:.4f}, "
                  f"kl={avg_kl:.4f}, gate={self.gate.item():.4f}")
            
            if wandb_run:
                wandb_run.log({"epoch": epoch+1, "train/loss": avg_loss,
                               "train/nll": avg_nll, "train/kl": avg_kl,
                               "train/gate": self.gate.item(),
                               "train/lr": scheduler.get_last_lr()[0]})
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            torch.cuda.empty_cache()
            gc.collect()
        
        self.trained = True
        print(f"Training complete. Gate={self.gate.item():.4f}")
    
    def eval(self, test_samples, max_new_tokens=512, num_beams=3, use_early_stopping=False):
        """Evaluate with z* = mu_phi(Q) from prior."""
        if not self.trained:
            print("Warning: ABC Vector not trained!")
        
        self.prior_net.eval()
        self.posterior_net.eval()
        
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        
        # Use model-specific terminators
        eos_ids = get_terminators(self.tokenizer, self.model_wrapper.model_name)
        
        from transformers import GenerationConfig
        gen_kwargs = {
            "max_new_tokens": max_new_tokens, "num_beams": num_beams,
            "do_sample": False, "temperature": 1.0, "top_p": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_ids,
        }
        if num_beams > 1:
            gen_kwargs["length_penalty"] = 0.0
        generation_config = GenerationConfig(**gen_kwargs)
        
        correct = 0
        total = len(test_samples)
        results = []
        
        pbar = tqdm(test_samples, desc=f"ABC Eval (L{self.layer_idx})", ncols=100)
        for sample in pbar:
            try:
                # Build prompts with chat template
                if self.dataset_type == "mmlu_pro":
                    question_raw = self.prompt_template["non_cot"].format(
                        question=sample.question, choices=sample.choices)
                    gen_raw = self.prompt_template["cot"].format(
                        question=sample.question, choices=sample.choices)
                else:
                    question_raw = self.prompt_template["non_cot"].format(
                        question=sample.question)
                    gen_raw = self.prompt_template["cot"].format(
                        question=sample.question)
                
                model_name = self.model_wrapper.model_name
                if needs_chat_template(model_name):
                    question_prompt = apply_chat_template_nothink(
                        question_raw, self.tokenizer, model_name, self.dataset_type)
                    gen_prompt = apply_chat_template(
                        gen_raw, self.tokenizer, model_name, self.dataset_type)
                else:
                    question_prompt = question_raw
                    gen_prompt = gen_raw
                
                # Encode question for r_Q
                q_enc = self.tokenizer(
                    question_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                q_ids = q_enc["input_ids"].to(target_device)
                q_mask = q_enc["attention_mask"].to(target_device)
                
                with torch.no_grad():
                    r_Q = self._extract_question_repr(q_ids, q_mask)
                    mu_phi, _ = self.prior_net(r_Q)
                    z_star = mu_phi
                
                gated_z_star = self.gate * z_star.squeeze(0)
                
                gen_enc = self.tokenizer(
                    gen_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
                gen_ids = gen_enc["input_ids"].to(target_device)
                gen_mask = gen_enc["attention_mask"].to(target_device)
                input_len = gen_ids.shape[1]
                
                self.model_wrapper.clear_hooks()
                self.model_wrapper.register_injection_hook(
                    self.layer_idx, vector=gated_z_star, scaling_factor=1.0, requires_grad=False)
                
                with torch.no_grad():
                    outputs = self.model_wrapper.model.generate(
                        gen_ids, attention_mask=gen_mask, generation_config=generation_config)
                
                generated_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
                predicted = extract_answer_from_text(generated_text, self.dataset_type)
                is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
                
                self.model_wrapper.clear_hooks()
                
                results.append({"predicted": predicted, "ground_truth": sample.answer,
                                "correct": is_correct, "generated_text": generated_text,
                                "num_tokens": len(outputs[0]) - input_len})
                
                if is_correct:
                    correct += 1
                pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%"})
                
            except Exception as e:
                print(f"\n  Error: {e}")
                results.append({"predicted": None, "ground_truth": sample.answer,
                                "correct": False, "error": str(e)})
        
        self.prior_net.train()
        self.posterior_net.train()
        return {"accuracy": correct / total * 100, "correct": correct, "total": total, "results": results}
    
    def get_vector(self):
        return None
    
    def get_state_dict(self):
        return {
            "prior": self.prior_net.state_dict(),
            "posterior": self.posterior_net.state_dict(),
            "gate": self.gate.detach().cpu(),
            "layer_idx": self.layer_idx,
            "abc_hidden_dim": self.abc_hidden_dim,
            "kl_beta": self.kl_beta,
            "kl_warmup_steps": self.kl_warmup_steps,
            "sigma_min": self.sigma_min,
        }
    
    def load_state_dict(self, state_dict, device=None):
        self.prior_net.load_state_dict(state_dict["prior"])
        self.posterior_net.load_state_dict(state_dict["posterior"])
        self.gate.data = state_dict["gate"]
        if device is not None:
            self._move_networks_to_device(device)
        self.trained = True
