"""Argument parser for ABC CoT Vector experiments."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ABC CoT Vector")

    # ==================== General Configuration ====================
    parser.add_argument(
        "--model_path", type=str,
        # default="/home/disk1/zby/ABCV/models/Meta-Llama-3.1-8B-Instruct",
        default="/home/disk1/zby/ABCV/models/Qwen2.5-Math-7B",
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen",
        choices=["qwen", "qwen3", "llama"],
        help="Model type: qwen (Qwen2.5-Math, plain text), "
             "qwen3 (Qwen3-8B, ChatML template), "
             "llama (Llama-3-Instruct, Llama template)"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="/home/disk1/zby/ABCV/data",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Base directory to save outputs (vectors saved to output_dir/{dataset}/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    # ==================== Method Selection ====================
    parser.add_argument("--method", type=str, default="abc", choices=["abc"])
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["train", "eval", "both"],
        help="Operation mode"
    )

    # ==================== Dataset Configuration ====================
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_support_samples", type=int, default=3000,
        help="Number of support samples for vector extraction/training"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=1000,
        help="Number of test samples for evaluation"
    )

    # ==================== CoT Vector Configuration ====================
    parser.add_argument(
        "--layer_idx", type=int, default=0,
        help="Layer index to inject/extract CoT Vector"
    )
    parser.add_argument(
        "--scaling_factor", type=float, default=1.0,
        help="Scaling factor for extracted vectors"
    )

    # ==================== Optimization Configuration ====================
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=None)

    # ==================== ABC Vector Configuration ====================
    parser.add_argument("--abc_hidden_dim", type=int, default=512)
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument("--kl_warmup_steps", type=int, default=500)
    parser.add_argument("--sigma_min", type=float, default=1e-4)
    parser.add_argument("--abc_learning_rate", type=float, default=5e-4)
    parser.add_argument("--g_init", type=float, default=0.0)

    # ==================== Generation Configuration ====================
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--use_early_stopping", action="store_true", default=False)

    # ==================== Logging Configuration ====================
    parser.add_argument("--skip_baseline", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=10)

    # ==================== Vector I/O ====================
    parser.add_argument("--save_vector", action="store_true", default=True)
    parser.add_argument("--abc_checkpoint_path", type=str, default=None)

    # ==================== Layer Sweep Configuration ====================
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--layer_step", type=int, default=2)
    parser.add_argument("--baseline_accuracy", type=float, default=None)
    parser.add_argument("--load_vectors_dir", type=str, default=None)

    args = parser.parse_args()

    if args.max_length is None:
        if args.dataset in {"gsm8k", "mmlu_pro"}:
            args.max_length = 1024
        else:
            args.max_length = 2048

    return args
