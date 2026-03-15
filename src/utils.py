"""
Utility functions for CoT Vectors reproduction.
Includes answer extraction (with Qwen3 thinking block support),
seeding, logging, and common helpers.
"""

import os
import re
import random
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import yaml
import numpy as np
import torch
from datetime import datetime


def setup_logging(output_dir: str, debug: bool = False) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_{timestamp}.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger("cot_vectors")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.getLogger("cot_vectors").info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def setup_wandb(
    args=None, config_path: str = "config/secrets.yaml",
    project: Optional[str] = None, run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None, enabled: bool = True
) -> Optional[Any]:
    if args is not None:
        project = project or getattr(args, 'wandb_project', None)
        config = config or vars(args)
        enabled = getattr(args, 'use_wandb', enabled)
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        secrets = load_config(config_path)
        wandb_config = secrets.get("wandb", {})
        api_key = wandb_config.get("api_key", os.environ.get("WANDB_API_KEY"))
        entity = wandb_config.get("entity")
        default_project = wandb_config.get("project", "cot-vectors-variational")
        if api_key and api_key != "PUT_KEY_HERE":
            os.environ["WANDB_API_KEY"] = api_key
        use_entity = entity if entity and entity != "your-username" else None
        run = wandb.init(
            project=project or default_project, entity=use_entity,
            name=run_name, config=config, reinit=True
        )
        logging.getLogger("cot_vectors").info(f"WandB initialized: {run.url}")
        return run
    except Exception as e:
        logging.getLogger("cot_vectors").warning(f"WandB init failed: {e}")
        return None


def save_vector(vector: torch.Tensor, output_path: str, metadata: Optional[Dict[str, Any]] = None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector.cpu(), "metadata": metadata or {}}, output_path)


def load_vector(vector_path: str) -> tuple:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    save_dict = torch.load(vector_path, map_location="cpu", weights_only=False)
    return save_dict["vector"], save_dict.get("metadata", {})


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    if num >= 1e9: return f"{num/1e9:.1f}B"
    elif num >= 1e6: return f"{num/1e6:.1f}M"
    elif num >= 1e3: return f"{num/1e3:.1f}K"
    return str(num)


def print_results_summary(
    model_name, method, layer_idx, dataset, accuracy,
    wandb_url=None, num_params=None
):
    print("\n" + "=" * 48)
    print("Results Summary")
    print("-" * 48)
    print(f"Model:       {model_name}")
    print(f"Method:      {method.upper()} CoT Vector")
    print(f"Layer:       {layer_idx}")
    print(f"Dataset:     {dataset.upper()}")
    print(f"Accuracy:    {accuracy:.2f}%")
    if num_params is not None:
        print(f"#Params:     {format_number(num_params)}")
    if wandb_url:
        print(f"WandB Run:   {wandb_url}")
    print("=" * 48 + "\n")


# ==================== Qwen3 Thinking Block Handling ====================

def strip_thinking_blocks(text: str) -> str:
    """
    Strip Qwen3's <think>...</think> reasoning blocks from generated text.
    
    Qwen3 with enable_thinking=True outputs:
        <think>
        Let me solve this step by step...
        </think>
        
        The answer is \\boxed{42}
    
    This function removes the thinking block, returning only the final answer part.
    Handles edge cases: incomplete blocks, nested content, multiple blocks,
    and the case where <think> tags were already stripped by skip_special_tokens.
    """
    if not text:
        return text
    
    # Remove complete <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Handle incomplete thinking block (model cut off before </think>)
    # If there's an opening <think> with no closing tag, remove from <think> to end
    if '<think>' in cleaned and '</think>' not in cleaned:
        cleaned = cleaned[:cleaned.index('<think>')]
    
    # Handle case where </think> appears without <think>
    # (e.g. <think> was at the very start and got stripped as special token)
    if '</think>' in cleaned and '<think>' not in cleaned:
        # Keep only the part after </think>
        cleaned = cleaned[cleaned.index('</think>') + len('</think>'):]
    
    # Clean up extra whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def clean_qwen3_output(text: str) -> str:
    """
    Clean Qwen3 raw output decoded with skip_special_tokens=False.
    
    Removes HF special tokens like <|im_end|>, <|endoftext|>, etc.
    but preserves <think> and </think> tags for strip_thinking_blocks().
    """
    if not text:
        return text
    # Remove tokens matching <|...|> pattern (HF special tokens)
    cleaned = re.sub(r'<\|[^>]+\|>', '', text)
    return cleaned.strip()


# ==================== Answer Extraction ====================

def extract_answer_from_text(text: str, dataset: str = "gsm8k") -> Optional[str]:
    """
    Extract the final answer from generated text.
    
    Handles Qwen3 thinking blocks by stripping them first.
    """
    if not text:
        return None
    
    # Strip Qwen3 thinking blocks before extraction
    text = strip_thinking_blocks(text)
    text = text.strip()
    
    if not text:
        return None
    
    # ========== Priority 1: Explicit answer formats ==========
    
    # \boxed{} format (highest priority)
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        answer = boxed_matches[-1].strip()
        if dataset == "mmlu_pro":
            letter_match = re.search(r"([A-J])", answer)
            if letter_match:
                return letter_match.group(1)
        return answer
    
    # #### X format (GSM8K style)
    hash_matches = re.findall(r"####\s*(.+?)(?:\n|$)", text)
    if hash_matches:
        return hash_matches[-1].strip()
    
    # "The answer is X" format (multiple variations)
    answer_patterns = [
        r"[Tt]he (?:final )?answer is[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Aa]nswer[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Tt]herefore,?\s+(?:the\s+)?(?:total\s+)?(?:answer\s+)?(?:is\s+)?\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Ss]o,?\s+(?:the\s+)?(?:total\s+)?(?:answer\s+)?(?:is\s+)?\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].replace(",", "").strip()
            if answer:
                return answer
    
    # ========== Priority 2: Dataset-specific patterns ==========
    
    if dataset == "mmlu_pro":
        patterns = [
            r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-J])\)?",
            r"\(?([A-J])\)?\s*(?:is correct|is the (?:correct |right )?answer)",
            r"^\s*\(?([A-J])\)?\s*$",
            r"\b([A-J])\b(?:\s*[.)]|\s*$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    
    # ========== Priority 3: Final answer near end of text ==========
    
    if dataset in ["gsm8k", "math_easy", "math_hard"]:
        text_end = text[-500:] if len(text) > 500 else text
        
        final_patterns = [
            r"=\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
            r"(?:pays?|costs?|receives?|earns?|gets?|made|makes?|spent|spends?|saved?|saves?|is|was|are|were|equals?)\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
            r"[Tt]otal[:\s=]+\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*[.!]",
            r"([0-9,]+(?:\.[0-9]+)?)\s*dollars",
        ]
        for pattern in final_patterns:
            matches = re.findall(pattern, text_end, re.IGNORECASE)
            if matches:
                return matches[-1].replace(",", "")
        
        text_very_end = text[-300:] if len(text) > 300 else text
        all_numbers = re.findall(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)(?![0-9])", text_very_end)
        if all_numbers:
            candidates = []
            for num_str in all_numbers:
                try:
                    num = float(num_str)
                    if 0 <= num <= 100000:
                        candidates.append(num_str)
                except:
                    pass
            if candidates:
                return candidates[-1]
    
    return None


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.strip().replace(",", "").replace("$", "").replace("%", "").lower()
    num_match = re.search(r"-?\d+\.?\d*", answer)
    if num_match:
        return num_match.group(0)
    return answer


def compare_answers(pred: str, gold: str, dataset: str = "gsm8k") -> bool:
    if pred is None:
        return False
    
    if dataset == "mmlu_pro":
        pred_letter = pred.strip().upper() if pred else ""
        gold_letter = gold.strip().upper() if gold else ""
        pred_match = re.search(r"([A-J])", pred_letter)
        gold_match = re.search(r"([A-J])", gold_letter)
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)
        return pred_letter == gold_letter
    
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm == gold_norm:
        return True
    
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


class AverageMeter:
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
