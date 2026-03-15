#!/usr/bin/env python3
"""
Main entry point for CoT Vectors experiments.

Supports: qwen (Qwen2.5-Math), qwen3 (Qwen3-8B), llama (Llama-3-Instruct)
Methods:  extracted, learnable, ua, abc

Usage:
    # Extracted CoT Vector (Qwen2.5-Math, GSM8K, layer 15)
    python main.py --model_name qwen --method extracted --mode both --layer_idx 15

    # Learnable CoT Vector (Qwen3-8B)
    python main.py --model_path /path/to/Qwen3-8B --model_name qwen3 \
        --method learnable --mode both --layer_idx 10

    # ABC Vector (Llama-3)
    python main.py --model_path /path/to/Llama-3-8B-Instruct --model_name llama \
        --method abc --mode both --layer_idx 12

    # Evaluation only (load pre-extracted vector)
    python main.py --model_name qwen --method extracted --mode eval \
        --vector_path ./outputs/gsm8k/extracted_vec_layer15.pt --layer_idx 15
"""

import os
import sys
import torch
import json
from datetime import datetime

from src.args import parse_args
from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import (
    setup_logging, set_seed, setup_wandb,
    save_vector, load_vector, print_results_summary,
)


def main():
    args = parse_args()
    
    # Setup
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    set_seed(args.seed)
    
    logger.info(f"CoT Vectors Experiment")
    logger.info(f"  Model:   {args.model_path} ({args.model_name})")
    logger.info(f"  Method:  {args.method}")
    logger.info(f"  Mode:    {args.mode}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Layer:   {args.layer_idx}")
    
    # WandB
    wandb_run = setup_wandb(args=args) if args.use_wandb else None
    
    # Load model & tokenizer
    logger.info("Loading model and tokenizer...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    logger.info(f"  Layers: {model_wrapper.num_layers}, Hidden: {model_wrapper.hidden_size}")
    
    # Load data
    test_samples = None
    if args.mode in ("eval", "both"):
        test_samples = load_dataset(
            args.data_path, args.dataset, "test",
            args.num_test_samples, args.seed,
        )
        logger.info(f"Loaded {len(test_samples)} test samples")
    
    support_samples = None
    if args.mode in ("extract", "train", "both"):
        support_samples = load_dataset(
            args.data_path, args.dataset, "train",
            args.num_support_samples, args.seed,
        )
        logger.info(f"Loaded {len(support_samples)} support samples")
    
    # ---- Baseline evaluation ----
    baseline_acc = args.baseline_accuracy
    if test_samples and not args.skip_baseline and baseline_acc is None:
        logger.info("Running baseline evaluation (CoT prompting, no vector)...")
        baseline_result = run_baseline_evaluation(
            model_wrapper, tokenizer, test_samples, args.dataset,
            args.max_new_tokens, args.num_beams, args.use_early_stopping,
        )
        baseline_acc = baseline_result["accuracy"]
        logger.info(f"Baseline accuracy: {baseline_acc:.2f}%")
        if wandb_run:
            wandb_run.log({"baseline_accuracy": baseline_acc})
    
    # ---- Method-specific logic ----
    vector = None
    
    if args.method == "extracted":
        from src.methods.extracted import ExtractedCoTVector
        method = ExtractedCoTVector(
            model_wrapper, tokenizer, args.layer_idx, args.dataset,
        )
        
        if args.vector_path and os.path.exists(args.vector_path):
            vector, metadata = load_vector(args.vector_path)
            logger.info(f"Loaded vector from {args.vector_path}")
        elif args.mode in ("extract", "both"):
            vector = method.extract(support_samples)
            if args.save_vector:
                vec_path = os.path.join(output_dir, f"extracted_vec_layer{args.layer_idx}.pt")
                save_vector(vector, vec_path, {
                    "method": "extracted", "layer_idx": args.layer_idx,
                    "model": args.model_name, "dataset": args.dataset,
                    "num_samples": len(support_samples),
                })
                logger.info(f"Saved vector to {vec_path}")
    
    elif args.method == "learnable":
        from src.methods.learnable import LearnableCoTVector
        method = LearnableCoTVector(
            model_wrapper, tokenizer, args.layer_idx, args.dataset,
            lambda_val=args.lambda_val, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            num_epochs=args.num_epochs, batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
        )
        
        if args.vector_path and os.path.exists(args.vector_path):
            vector, metadata = load_vector(args.vector_path)
            logger.info(f"Loaded vector from {args.vector_path}")
        elif args.mode in ("train", "both"):
            vector = method.train(support_samples, wandb_run=wandb_run)
            if args.save_vector:
                vec_path = os.path.join(output_dir, f"learnable_vec_layer{args.layer_idx}.pt")
                save_vector(vector, vec_path, {
                    "method": "learnable", "layer_idx": args.layer_idx,
                    "model": args.model_name, "dataset": args.dataset,
                    "lambda_val": args.lambda_val, "lr": args.learning_rate,
                    "epochs": args.num_epochs,
                })
                logger.info(f"Saved vector to {vec_path}")
    
    elif args.method == "ua":
        from src.methods.ua_vector import UACoTVector
        method = UACoTVector(
            model_wrapper, tokenizer, args.layer_idx, args.dataset,
            tau_squared=args.tau_squared, min_variance=args.min_variance,
        )
        
        if args.vector_path and os.path.exists(args.vector_path):
            vector, metadata = load_vector(args.vector_path)
            logger.info(f"Loaded vector from {args.vector_path}")
        elif args.mode in ("extract", "both"):
            vector = method.extract(support_samples)
            if args.save_vector:
                vec_path = os.path.join(output_dir, f"ua_vec_layer{args.layer_idx}.pt")
                save_vector(vector, vec_path, {
                    "method": "ua", "layer_idx": args.layer_idx,
                    "model": args.model_name, "dataset": args.dataset,
                    "tau_squared": args.tau_squared,
                    "num_samples": len(support_samples),
                })
                logger.info(f"Saved vector to {vec_path}")
    
    elif args.method == "abc":
        from src.methods.abc_vector import ABCCoTVector
        method = ABCCoTVector(
            model_wrapper, tokenizer, args.layer_idx, args.dataset,
            abc_hidden_dim=args.abc_hidden_dim, kl_beta=args.kl_beta,
            kl_warmup_steps=args.kl_warmup_steps, sigma_min=args.sigma_min,
            learning_rate=args.abc_learning_rate, weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio, num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
        )
        
        if args.abc_checkpoint_path and os.path.exists(args.abc_checkpoint_path):
            state = torch.load(args.abc_checkpoint_path, map_location="cpu", weights_only=False)
            method.load_state_dict(state, device=model_wrapper.device)
            logger.info(f"Loaded ABC checkpoint from {args.abc_checkpoint_path}")
        elif args.mode in ("train", "both"):
            method.train(support_samples, wandb_run=wandb_run)
            if args.save_vector:
                ckpt_path = os.path.join(output_dir, f"abc_ckpt_layer{args.layer_idx}.pt")
                torch.save(method.get_state_dict(), ckpt_path)
                logger.info(f"Saved ABC checkpoint to {ckpt_path}")
        
        # ABC has its own eval loop
        if test_samples and args.mode in ("eval", "both"):
            abc_result = method.eval(
                test_samples, args.max_new_tokens, args.num_beams, args.use_early_stopping,
            )
            abc_acc = abc_result["accuracy"]
            logger.info(f"ABC Vector accuracy: {abc_acc:.2f}%")
            print_results_summary(
                args.model_name, args.method, args.layer_idx,
                args.dataset, abc_acc,
            )
            if wandb_run:
                wandb_run.log({"abc_accuracy": abc_acc})
            
            # Save results
            result_path = os.path.join(output_dir, f"abc_results_layer{args.layer_idx}.json")
            with open(result_path, "w") as f:
                json.dump({
                    "method": "abc", "layer_idx": args.layer_idx,
                    "accuracy": abc_acc, "baseline_accuracy": baseline_acc,
                    "model": args.model_name, "dataset": args.dataset,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
            return  # ABC handles its own eval
    
    # ---- Injection evaluation (for extracted/learnable/ua) ----
    if vector is not None and test_samples and args.mode in ("eval", "both"):
        logger.info(f"Running injection evaluation at layer {args.layer_idx}...")
        inject_result = run_injection_evaluation(
            model_wrapper, tokenizer, test_samples, vector, args.layer_idx,
            args.dataset, args.scaling_factor, args.max_new_tokens,
            args.num_beams, args.use_early_stopping,
        )
        inject_acc = inject_result["accuracy"]
        logger.info(f"Injection accuracy: {inject_acc:.2f}%")
        
        print_results_summary(
            args.model_name, args.method, args.layer_idx,
            args.dataset, inject_acc,
        )
        
        if wandb_run:
            wandb_run.log({"injection_accuracy": inject_acc})
        
        # Save results
        result_path = os.path.join(
            output_dir, f"{args.method}_results_layer{args.layer_idx}.json"
        )
        with open(result_path, "w") as f:
            json.dump({
                "method": args.method, "layer_idx": args.layer_idx,
                "accuracy": inject_acc, "baseline_accuracy": baseline_acc,
                "scaling_factor": args.scaling_factor,
                "model": args.model_name, "dataset": args.dataset,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
    
    if wandb_run:
        wandb_run.finish()
    
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
