#!/usr/bin/env python3
"""Main entry point for ABC CoT Vector experiments."""

import json
import os
from datetime import datetime

import torch

from src.args import parse_args
from src.data_utils import load_dataset
from src.eval import run_baseline_evaluation
from src.methods.abc_vector import ABCCoTVector
from src.models import CoTModelWrapper, load_tokenizer
from src.utils import print_results_summary, set_seed, setup_logging


def run_abc_experiment(args) -> None:
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    set_seed(args.seed)

    logger.info("ABC CoT Vector Experiment")
    logger.info(f"  Model:   {args.model_path} ({args.model_name})")
    logger.info(f"  Mode:    {args.mode}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Layer:   {args.layer_idx}")
    logger.info(f"  MaxLen:  {args.max_length}")

    logger.info("Loading model and tokenizer...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    logger.info(f"  Layers: {model_wrapper.num_layers}, Hidden: {model_wrapper.hidden_size}")

    test_samples = None
    if args.mode in ("eval", "both"):
        test_samples = load_dataset(
            args.data_path,
            args.dataset,
            "test",
            args.num_test_samples,
            args.seed,
        )
        logger.info(f"Loaded {len(test_samples)} test samples")

    support_samples = None
    if args.mode in ("train", "both"):
        support_samples = load_dataset(
            args.data_path,
            args.dataset,
            "train",
            args.num_support_samples,
            args.seed,
        )
        logger.info(f"Loaded {len(support_samples)} support samples")

    baseline_acc = args.baseline_accuracy
    if test_samples and not args.skip_baseline and baseline_acc is None:
        logger.info("Running baseline evaluation (CoT prompting, no vector)...")
        baseline_result = run_baseline_evaluation(
            model_wrapper,
            tokenizer,
            test_samples,
            args.dataset,
            args.max_new_tokens,
            args.num_beams,
            args.use_early_stopping,
            args.max_length,
        )
        baseline_acc = baseline_result["accuracy"]
        logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")

    abc_method = ABCCoTVector(
        model_wrapper,
        tokenizer,
        args.layer_idx,
        args.dataset,
        abc_hidden_dim=args.abc_hidden_dim,
        kl_beta=args.kl_beta,
        kl_warmup_steps=args.kl_warmup_steps,
        sigma_min=args.sigma_min,
        learning_rate=args.abc_learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        g_init=args.g_init,
    )

    if args.abc_checkpoint_path and os.path.exists(args.abc_checkpoint_path):
        state = torch.load(args.abc_checkpoint_path, map_location="cpu", weights_only=False)
        abc_method.load_state_dict(state, device=model_wrapper.device)
        logger.info(f"Loaded ABC checkpoint from {args.abc_checkpoint_path}")
    elif args.mode in ("train", "both"):
        abc_method.train(support_samples)
        if args.save_vector:
            checkpoint_path = os.path.join(output_dir, f"abc_ckpt_layer{args.layer_idx}.pt")
            torch.save(abc_method.get_state_dict(), checkpoint_path)
            logger.info(f"Saved ABC checkpoint to {checkpoint_path}")

    if test_samples and args.mode in ("eval", "both"):
        abc_result = abc_method.eval(
            test_samples,
            args.max_new_tokens,
            args.num_beams,
            args.use_early_stopping,
            args.max_length,
        )
        abc_acc = abc_result["accuracy"]
        logger.info(f"ABC Vector accuracy: {abc_acc:.2f}%")
        print_results_summary(args.model_name, "abc", args.layer_idx, args.dataset, abc_acc)

        result_path = os.path.join(output_dir, f"abc_results_layer{args.layer_idx}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "abc",
                    "layer_idx": args.layer_idx,
                    "accuracy": abc_acc,
                    "baseline_accuracy": baseline_acc,
                    "model": args.model_name,
                    "dataset": args.dataset,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    logger.info("Experiment complete.")


def main() -> None:
    args = parse_args()
    run_abc_experiment(args)


if __name__ == "__main__":
    main()
