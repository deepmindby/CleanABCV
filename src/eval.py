"""
Evaluation logic for CoT Vectors.

Supports all models: qwen (plain text), qwen3 (ChatML), llama (Llama-3 template).

Key features:
1. apply_chat_template() for proper prompt formatting
2. get_terminators() for correct EOS token IDs
3. Auto max_new_tokens for Qwen3 thinking mode (needs 2048+)
4. Decode with skip_special_tokens=False for Qwen3 to preserve <think> tags
5. strip_thinking_blocks() to extract clean answer text
"""

import torch
import re
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .models import CoTModelWrapper, load_tokenizer
from .data_utils import PROMPT_TEMPLATES, apply_chat_template, get_terminators
from .utils import extract_answer_from_text, compare_answers, strip_thinking_blocks


# ==================== Constants ====================

# Qwen3 thinking mode easily produces 1000+ tokens of reasoning before the answer.
# 512 is guaranteed to truncate mid-think, yielding pred=None on every sample.
QWEN3_MIN_NEW_TOKENS = 2048

# Models that have thinking mode (output <think>...</think> blocks)
THINKING_MODELS = {"qwen3"}


# ==================== Stopping Criteria ====================

class AnswerStoppingCriteria(StoppingCriteria):
    """Stop generation when answer pattern is detected after thinking."""

    def __init__(self, tokenizer, dataset_type: str = "gsm8k",
                 model_name: str = "qwen", min_tokens: int = 30):
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.model_name = model_name
        self.min_tokens = min_tokens
        self.generated_tokens = 0

        if dataset_type == "mmlu_pro":
            self.patterns = [
                re.compile(r'\\boxed\s*\{[A-J]\}'),
                re.compile(r'[Tt]he\s+answer\s+is\s*:?\s*\(?([A-J])\)?'),
            ]
        else:
            self.patterns = [
                re.compile(r'\\boxed\s*\{[^{}]+\}'),
                re.compile(r'####\s*[\d,]+'),
            ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        self.generated_tokens += 1
        if self.generated_tokens < self.min_tokens:
            return False

        # For thinking models, don't stop until we've exited the think block
        text = self.tokenizer.decode(input_ids[0, -200:], skip_special_tokens=False)
        if self.model_name in THINKING_MODELS:
            # Still inside <think> block — don't stop
            if '<think>' in text and '</think>' not in text:
                return False

        return any(p.search(text) for p in self.patterns)

    def reset(self):
        self.generated_tokens = 0


# ==================== Evaluator ====================

class CoTEvaluator:
    """Evaluator for CoT Vector experiments."""

    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        dataset_type: str = "gsm8k",
        max_new_tokens: int = 512,
        num_beams: int = 3,
        temperature: float = 1.0,
        do_sample: bool = False,
        use_early_stopping: bool = False,
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        self.is_thinking_model = model_wrapper.model_name in THINKING_MODELS

        # ---- Auto-adjust max_new_tokens for thinking models ----
        if self.is_thinking_model and max_new_tokens < QWEN3_MIN_NEW_TOKENS:
            print(f"  ⚠ Qwen3 thinking mode: auto-increasing max_new_tokens "
                  f"from {max_new_tokens} → {QWEN3_MIN_NEW_TOKENS}")
            max_new_tokens = QWEN3_MIN_NEW_TOKENS
        self.max_new_tokens = max_new_tokens

        # Early stopping (recommended for Qwen3 to save time)
        self.stopping_criteria = None
        if use_early_stopping or self.is_thinking_model:
            if self.is_thinking_model and not use_early_stopping:
                print(f"  ⚠ Qwen3: auto-enabling early stopping to avoid "
                      f"wasting tokens after answer")
            self.stopping_criteria = StoppingCriteriaList([
                AnswerStoppingCriteria(
                    tokenizer, dataset_type, model_wrapper.model_name)
            ])

        # Model-specific terminators
        eos_token_ids = get_terminators(tokenizer, model_wrapper.model_name)
        print(f"  EOS token IDs: {eos_token_ids}")
        print(f"  max_new_tokens: {max_new_tokens}")

        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": 1.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_ids,
        }
        if num_beams > 1:
            gen_kwargs["length_penalty"] = 0.0

        self.generation_config = GenerationConfig(**gen_kwargs)
        self.prompt_template = PROMPT_TEMPLATES.get(
            dataset_type, PROMPT_TEMPLATES["gsm8k"])

    def _decode_output(self, generated_ids: torch.Tensor) -> str:
        """Decode generated token ids, handling thinking model output properly."""
        if self.is_thinking_model:
            # Decode WITHOUT stripping special tokens so <think>/</ think> are preserved
            raw_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=False)
            # Remove HF special tokens like <|im_end|>, <|endoftext|>
            # but keep <think> and </think> for strip_thinking_blocks()
            raw_text = re.sub(r'<\|[^>]+\|>', '', raw_text)
            return raw_text
        else:
            return self.tokenizer.decode(
                generated_ids, skip_special_tokens=True)

    def evaluate_sample(
        self,
        sample,
        vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        scaling_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate a single sample with model-specific prompt formatting."""

        # Build raw prompt
        if self.dataset_type == "mmlu_pro":
            raw_prompt = self.prompt_template["cot"].format(
                question=sample.question, choices=sample.choices
            )
        else:
            raw_prompt = self.prompt_template["cot"].format(
                question=sample.question)

        # Apply model-specific chat template
        prompt = apply_chat_template(
            raw_prompt, self.tokenizer,
            self.model_wrapper.model_name, self.dataset_type,
        )

        # Tokenize
        device = self.model_wrapper.device
        encoding = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        input_len = input_ids.shape[1]

        # Clear previous hooks
        self.model_wrapper.clear_hooks()

        # Register injection hook if vector provided
        if vector is not None and layer_idx is not None:
            self.model_wrapper.register_injection_hook(
                layer_idx, vector, scaling_factor)

        # Reset stopping criteria
        if self.stopping_criteria:
            for sc in self.stopping_criteria:
                if hasattr(sc, 'reset'):
                    sc.reset()

        # Generate
        with torch.no_grad():
            outputs = self.model_wrapper.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                stopping_criteria=self.stopping_criteria,
            )

        # Decode output
        generated_ids = outputs[0, input_len:]
        raw_text = self._decode_output(generated_ids)

        # Extract answer (strip_thinking_blocks is called inside)
        predicted = extract_answer_from_text(raw_text, self.dataset_type)
        is_correct = compare_answers(predicted, sample.answer, self.dataset_type)

        self.model_wrapper.clear_hooks()

        return {
            "predicted": predicted,
            "ground_truth": sample.answer,
            "correct": is_correct,
            "generated_text": raw_text,
            "num_tokens": len(generated_ids),
        }

    def evaluate_dataset(
        self,
        samples: List,
        vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        scaling_factor: float = 1.0,
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        correct = 0
        total = len(samples)
        results = []

        pbar = tqdm(samples, desc=desc, ncols=100)
        for sample in pbar:
            result = self.evaluate_sample(
                sample, vector, layer_idx, scaling_factor)
            results.append(result)
            if result["correct"]:
                correct += 1
            acc = correct / len(results) * 100
            pbar.set_postfix({"acc": f"{acc:.1f}%"})

        accuracy = correct / total * 100
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }


# ==================== Convenience functions ====================

def run_baseline_evaluation(
    model_wrapper, tokenizer, test_samples, dataset_type,
    max_new_tokens=512, num_beams=3, use_early_stopping=False,
):
    """Run baseline evaluation (CoT prompting, no vector injection)."""
    evaluator = CoTEvaluator(
        model_wrapper=model_wrapper, tokenizer=tokenizer,
        dataset_type=dataset_type, max_new_tokens=max_new_tokens,
        num_beams=num_beams, use_early_stopping=use_early_stopping,
    )
    return evaluator.evaluate_dataset(test_samples, desc="Baseline")


def run_injection_evaluation(
    model_wrapper, tokenizer, test_samples, vector, layer_idx,
    dataset_type, scaling_factor=1.0, max_new_tokens=512,
    num_beams=3, use_early_stopping=False,
):
    """Run evaluation with CoT vector injection."""
    evaluator = CoTEvaluator(
        model_wrapper=model_wrapper, tokenizer=tokenizer,
        dataset_type=dataset_type, max_new_tokens=max_new_tokens,
        num_beams=num_beams, use_early_stopping=use_early_stopping,
    )
    return evaluator.evaluate_dataset(
        test_samples, vector=vector, layer_idx=layer_idx,
        scaling_factor=scaling_factor, desc=f"Injection (L{layer_idx})",
    )
