"""
Data loading and processing utilities for CoT Vectors.
Handles GSM8K, MATH, and MMLU-Pro datasets with proper prompt templates.

Supports model-specific chat templates:
- qwen: Qwen2.5-Math-7B (plain text, no template needed)
- qwen3: Qwen3-8B (ChatML format with <|im_start|>/<|im_end|>)
- llama: Llama-3-8B-Instruct (<|start_header_id|>/<|eot_id|>)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


logger = logging.getLogger("cot_vectors")


# ==================== Prompt Templates (Table 3 in Appendix) ====================

PROMPT_TEMPLATES = {
    "gsm8k": {
        "system": (
            "You are a helpful and precise assistant for solving math problems."
        ),
        "cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Put your answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
    },
    "math": {
        "system": (
            "You are a helpful and precise assistant for solving math problems."
        ),
        "cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Put your answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
    },
    "mmlu_pro": {
        "system": (
            "You are a helpful and precise assistant for solving problems."
        ),
        "cot": (
            "You are a helpful and precise assistant for solving problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}. "
            "Your final output should be only the uppercase letter of the correct choice (e.g., A).\n\n"
            "Question: {question}\n"
            "Choices:\n{choices}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving problems. "
            "Put your answer within \\boxed{{}}. "
            "Your final output should be only the uppercase letter of the correct choice (e.g., A).\n\n"
            "Question: {question}\n"
            "Choices:\n{choices}\n"
        ),
    },
}


# ==================== Model Detection ====================

# Models that need chat template wrapping (instruct/chat models)
# Qwen2.5-Math-7B is a specialized math model that works with plain text
NEEDS_CHAT_TEMPLATE = {"llama", "qwen3"}

# Models that DON'T need chat template (plain text is fine)
PLAIN_TEXT_MODELS = {"qwen"}  # Qwen2.5-Math-7B


def needs_chat_template(model_name: str) -> bool:
    """Check if the model requires chat template formatting."""
    return model_name in NEEDS_CHAT_TEMPLATE


# ==================== Model-Specific Formatting ====================

def apply_chat_template(
    prompt_text: str,
    tokenizer,
    model_name: str,
    dataset_type: str = "gsm8k",
) -> str:
    """
    Apply model-specific chat template to a prompt.
    
    Behavior by model_name:
    - "qwen":  Returns prompt as-is (Qwen2.5-Math works with plain text)
    - "qwen3": Wraps in ChatML format via tokenizer.apply_chat_template()
    - "llama": Wraps in Llama-3 format via tokenizer.apply_chat_template()
    
    Args:
        prompt_text: The raw prompt string (from PROMPT_TEMPLATES)
        tokenizer: The HuggingFace tokenizer
        model_name: Model type ("qwen", "qwen3", or "llama")
        dataset_type: Dataset type for system message selection
        
    Returns:
        Formatted prompt string
    """
    if not needs_chat_template(model_name):
        # Qwen2.5-Math-7B etc. work fine with plain text prompts
        return prompt_text
    
    # ---- Build messages list for chat template ----
    templates = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    system_msg = templates.get("system", "You are a helpful assistant.")
    
    # Extract user message: everything starting from "Question:"
    if "Question:" in prompt_text:
        q_idx = prompt_text.index("Question:")
        user_msg = prompt_text[q_idx:].strip()
        
        # Extract instruction part (between system msg and "Question:")
        pre_question = prompt_text[:q_idx].strip()
        instruction_part = pre_question
        for sys_text in [templates.get("system", "")]:
            instruction_part = instruction_part.replace(sys_text, "").strip()
        
        if instruction_part:
            system_msg = system_msg + " " + instruction_part
    else:
        user_msg = prompt_text.strip()
    
    # ---- Qwen3-specific: add /think or /no_think control ----
    # For evaluation prompts (CoT), we want the model to think naturally.
    # Qwen3 enables thinking by default, so no special tag needed.
    # For non-CoT prompts used in extraction, we could add /no_think,
    # but that's handled at the caller level.
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    # Use tokenizer's built-in chat template
    try:
        # Qwen3 and Llama-3 both support apply_chat_template
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        
        # Qwen3 supports enable_thinking parameter
        if model_name == "qwen3":
            # Enable thinking for CoT evaluation (model reasons internally)
            # The <think>...</think> block will be in the output
            kwargs["enable_thinking"] = True
        
        formatted = tokenizer.apply_chat_template(messages, **kwargs)
        return formatted
        
    except TypeError as e:
        # Older transformers version may not support enable_thinking
        if "enable_thinking" in str(e):
            logger.warning("tokenizer.apply_chat_template doesn't support enable_thinking, "
                          "retrying without it")
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return formatted
        raise
    except Exception as e:
        logger.warning(f"apply_chat_template failed ({e}), falling back to plain text")
        return prompt_text


def apply_chat_template_nothink(
    prompt_text: str,
    tokenizer,
    model_name: str,
    dataset_type: str = "gsm8k",
) -> str:
    """
    Apply chat template with thinking DISABLED (for Qwen3 non-CoT extraction).
    
    In CoT Vector extraction, the non-CoT path should not think.
    For Qwen3, this means using enable_thinking=False.
    For other models, same as apply_chat_template.
    """
    if not needs_chat_template(model_name):
        return prompt_text
    
    if model_name != "qwen3":
        return apply_chat_template(prompt_text, tokenizer, model_name, dataset_type)
    
    # Qwen3 with thinking disabled
    templates = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    system_msg = templates.get("system", "You are a helpful assistant.")
    
    if "Question:" in prompt_text:
        q_idx = prompt_text.index("Question:")
        user_msg = prompt_text[q_idx:].strip()
        pre_question = prompt_text[:q_idx].strip()
        instruction_part = pre_question
        for sys_text in [templates.get("system", "")]:
            instruction_part = instruction_part.replace(sys_text, "").strip()
        if instruction_part:
            system_msg = system_msg + " " + instruction_part
    else:
        user_msg = prompt_text.strip()
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return formatted
    except TypeError:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return formatted
    except Exception as e:
        logger.warning(f"apply_chat_template (no_think) failed ({e}), falling back")
        return prompt_text


def get_terminators(tokenizer, model_name: str) -> List[int]:
    """
    Get the correct EOS/terminator token IDs for the model.
    
    Model-specific terminators:
    - qwen:  [eos_token_id] (default, usually 151643)
    - qwen3: [eos_token_id, <|im_end|>] 
             <|im_end|> marks the end of an assistant turn in ChatML
    - llama: [eos_token_id, <|eot_id|> (128009)]
             <|eot_id|> marks end-of-turn in Llama-3
    
    Args:
        tokenizer: The HuggingFace tokenizer
        model_name: Model type ("qwen", "qwen3", or "llama")
        
    Returns:
        List of EOS token IDs
    """
    eos_ids = []
    
    # Always include the default EOS
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    
    if model_name == "llama":
        # Llama-3: need <|eot_id|> (128009)
        _try_add_special_token(tokenizer, eos_ids, "<|eot_id|>", fallback_ids=[128009])
    
    elif model_name == "qwen3":
        # Qwen3: need <|im_end|> (ChatML end marker)
        _try_add_special_token(tokenizer, eos_ids, "<|im_end|>")
        
        # Also add <|endoftext|> if different from eos_token_id
        _try_add_special_token(tokenizer, eos_ids, "<|endoftext|>")
    
    # model_name == "qwen": default eos_token_id is sufficient
    
    logger.info(f"Terminators for {model_name}: {eos_ids}")
    return eos_ids


def _try_add_special_token(
    tokenizer, 
    eos_ids: List[int], 
    token_str: str,
    fallback_ids: Optional[List[int]] = None,
):
    """Try to find a special token and add it to the EOS list."""
    try:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is not None and token_id != getattr(tokenizer, 'unk_token_id', None):
            if token_id not in eos_ids:
                eos_ids.append(token_id)
                logger.info(f"Added terminator: {token_str} -> {token_id}")
                return
    except Exception:
        pass
    
    # Fallback to known IDs
    if fallback_ids:
        for known_id in fallback_ids:
            if known_id not in eos_ids:
                eos_ids.append(known_id)
                logger.info(f"Added terminator (fallback): {token_str} -> {known_id}")


def build_prompt(
    dataset_type: str,
    question: str,
    model_name: str,
    tokenizer,
    cot: bool = True,
    choices: Optional[str] = None,
) -> str:
    """
    Build a properly formatted prompt for the given model and dataset.
    Convenience function that applies prompt template + chat template.
    """
    templates = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    prompt_key = "cot" if cot else "non_cot"
    
    if dataset_type == "mmlu_pro" and choices:
        raw_prompt = templates[prompt_key].format(question=question, choices=choices)
    else:
        raw_prompt = templates[prompt_key].format(question=question)
    
    return apply_chat_template(raw_prompt, tokenizer, model_name, dataset_type)


# ==================== Data Classes ====================

@dataclass
class CoTSample:
    """Data class for a single CoT sample."""
    question: str
    cot: Optional[str]
    answer: str
    full_cot_text: str
    full_non_cot_text: str
    choices: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CoTDataset(Dataset):
    """PyTorch Dataset for CoT Vector training/evaluation."""
    
    def __init__(
        self,
        samples: List[CoTSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        include_cot: bool = True
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_cot = include_cot
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        if self.include_cot:
            text = sample.full_cot_text
        else:
            text = sample.full_non_cot_text
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": sample.answer,
            "question": sample.question,
            "cot": sample.cot,
            "full_cot_text": sample.full_cot_text,
            "full_non_cot_text": sample.full_non_cot_text,
        }


def format_cot_sample(
    question: str,
    cot: Optional[str],
    answer: str,
    dataset_type: str,
    choices: Optional[str] = None
) -> CoTSample:
    """Format a sample into CoTSample with proper prompt templates."""
    templates = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    if dataset_type == "mmlu_pro" and choices:
        cot_prompt = templates["cot"].format(question=question, choices=choices)
        non_cot_prompt = templates["non_cot"].format(question=question, choices=choices)
    else:
        cot_prompt = templates["cot"].format(question=question)
        non_cot_prompt = templates["non_cot"].format(question=question)
    
    if cot:
        full_cot_text = f"{cot_prompt}\n{cot}\n\nThe answer is \\boxed{{{answer}}}"
    else:
        full_cot_text = f"{cot_prompt}\nThe answer is \\boxed{{{answer}}}"
    
    full_non_cot_text = f"{non_cot_prompt}\nThe answer is \\boxed{{{answer}}}"
    
    return CoTSample(
        question=question,
        cot=cot,
        answer=answer,
        full_cot_text=full_cot_text,
        full_non_cot_text=full_non_cot_text,
        choices=choices if dataset_type == "mmlu_pro" else None,
    )


# ==================== Dataset Loaders ====================

def load_gsm8k(
    data_path: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """Load GSM8K dataset."""
    logger.info(f"Loading GSM8K {split} split from {data_path}")
    
    possible_paths = [
        os.path.join(data_path, "gsm8k", f"{split}.jsonl"),
        os.path.join(data_path, f"gsm8k_{split}.jsonl"),
        os.path.join(data_path, "gsm8k", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"GSM8K data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get("question", data.get("problem", ""))
            answer = data.get("answer", "")
            
            if "####" in str(answer):
                parts = str(answer).split("####")
                cot = parts[0].strip() if len(parts) > 1 else None
                final_answer = parts[-1].strip()
            else:
                cot = data.get("cot", data.get("solution", None))
                final_answer = str(answer).strip()
            
            sample = format_cot_sample(
                question=question, cot=cot, answer=final_answer, dataset_type="gsm8k"
            )
            samples.append(sample)
    
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} GSM8K samples")
    return samples


def load_math(
    data_path: str,
    split: str = "train",
    difficulty: str = "all",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """Load MATH dataset."""
    logger.info(f"Loading MATH {split} split ({difficulty}) from {data_path}")
    
    possible_paths = [
        os.path.join(data_path, "math", f"{split}.jsonl"),
        os.path.join(data_path, f"math_{split}.jsonl"),
        os.path.join(data_path, "MATH", f"{split}.jsonl"),
        os.path.join(data_path, "math", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    use_train_for_test = False
    if file_path is None and split == "test":
        train_paths = [
            os.path.join(data_path, "math", "train.jsonl"),
            os.path.join(data_path, "math_train.jsonl"),
            os.path.join(data_path, "MATH", "train.jsonl"),
        ]
        for path in train_paths:
            if os.path.exists(path):
                file_path = path
                use_train_for_test = True
                break
    
    if file_path is None:
        raise FileNotFoundError(
            f"MATH data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            level = data.get("level", 3)
            if isinstance(level, str):
                try:
                    level = int(level.replace("Level ", "").strip())
                except ValueError:
                    level = 3
            
            if difficulty == "easy" and level > 3:
                continue
            elif difficulty == "hard" and level < 4:
                continue
            
            question = data.get("question", data.get("problem", ""))
            solution = data.get("cot", data.get("solution", ""))
            answer = data.get("answer", "")
            
            if not answer and solution:
                import re
                match = re.search(r"\\boxed\{([^}]+)\}", solution)
                if match:
                    answer = match.group(1)
            
            sample = format_cot_sample(
                question=question, cot=solution, answer=answer, dataset_type="math"
            )
            samples.append(sample)
    
    if use_train_for_test:
        import random
        random.seed(seed)
        random.shuffle(samples)
        test_size = max(len(samples) // 5, 200)
        samples = samples[-test_size:]
    
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} MATH samples")
    return samples


def load_mmlu_pro(
    data_path: str,
    split: str = "validation",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """Load MMLU-Pro dataset."""
    logger.info(f"Loading MMLU-Pro {split} split from {data_path}")
    
    possible_paths = [
        os.path.join(data_path, "mmlu_pro", f"{split}.jsonl"),
        os.path.join(data_path, f"mmlu_pro_{split}.jsonl"),
        os.path.join(data_path, "MMLU-Pro", f"{split}.jsonl"),
        os.path.join(data_path, "mmlu_pro", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"MMLU-Pro data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get("question", "")
            choices = data.get("choices", data.get("options", []))
            answer = data.get("answer", "")
            cot = data.get("cot", data.get("rationale", data.get("explanation", "")))
            
            if isinstance(answer, int):
                answer = chr(65 + answer)
            elif isinstance(answer, str) and answer.isdigit():
                answer = chr(65 + int(answer))
            
            if isinstance(choices, list):
                choices_text = "\n".join([
                    f"{chr(65+i)}. {choice}" 
                    for i, choice in enumerate(choices)
                ])
            else:
                choices_text = str(choices)
            
            sample = format_cot_sample(
                question=question, cot=cot, answer=str(answer),
                dataset_type="mmlu_pro", choices=choices_text
            )
            samples.append(sample)
    
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} MMLU-Pro samples")
    return samples


def load_dataset(
    data_path: str,
    dataset_name: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """Load a dataset by name."""
    loaders = {
        "gsm8k": lambda: load_gsm8k(data_path, split, num_samples, seed),
        "math_easy": lambda: load_math(data_path, split, "easy", num_samples, seed),
        "math_hard": lambda: load_math(data_path, split, "hard", num_samples, seed),
        "math": lambda: load_math(data_path, split, "all", num_samples, seed),
        "mmlu_pro": lambda: load_mmlu_pro(
            data_path, 
            "validation" if split == "train" else split,
            num_samples, seed
        ),
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]()


def create_dataloader(
    samples: List[CoTSample],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 2048,
    include_cot: bool = True,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader from samples."""
    dataset = CoTDataset(
        samples=samples, tokenizer=tokenizer,
        max_length=max_length, include_cot=include_cot
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )


def collate_for_extraction(
    batch: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict[str, Any]:
    """Custom collate function for CoT vector extraction."""
    cot_texts = [item["full_cot_text"] for item in batch]
    non_cot_texts = [item["full_non_cot_text"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    cot_encoding = tokenizer(
        cot_texts, max_length=max_length, padding=True,
        truncation=True, return_tensors="pt"
    )
    non_cot_encoding = tokenizer(
        non_cot_texts, max_length=max_length, padding=True,
        truncation=True, return_tensors="pt"
    )
    
    return {
        "cot_input_ids": cot_encoding["input_ids"],
        "cot_attention_mask": cot_encoding["attention_mask"],
        "non_cot_input_ids": non_cot_encoding["input_ids"],
        "non_cot_attention_mask": non_cot_encoding["attention_mask"],
        "answers": answers,
        "cot_texts": cot_texts,
        "non_cot_texts": non_cot_texts,
    }


if __name__ == "__main__":
    print("Testing data utilities...")
    
    sample = format_cot_sample(
        question="What is 2 + 2?",
        cot="Let me think step by step. 2 + 2 equals 4.",
        answer="4",
        dataset_type="gsm8k"
    )
    print(f"Sample question: {sample.question}")
    print(f"Sample answer: {sample.answer}")
    
    print("\nData utilities test complete!")
