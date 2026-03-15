#!/usr/bin/env python3
"""Layer sweep script for ABC CoT Vector."""

import os
from datetime import datetime

import torch
from tqdm import tqdm

from src.args import parse_args
from src.data_utils import load_dataset
from src.eval import run_baseline_evaluation
from src.methods.abc_vector import ABCCoTVector
from src.models import CoTModelWrapper, load_tokenizer
from src.utils import set_seed


def get_output_dir(base_dir: str, dataset: str) -> str:
    output_dir = os.path.join(base_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = get_output_dir(args.output_dir, args.dataset)

    print("=" * 70)
    print("ABC CoT Vector Layer Sweep")
    print("=" * 70)
    print(f"Model:    {args.model_path.split('/')[-1]}")
    print(f"Method:   abc")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {output_dir}")
    print(f"Support:  {args.num_support_samples}, Test: {args.num_test_samples}")
    print(f"Skip baseline: {args.skip_baseline}")
    print(f"ABC Config: hidden_dim={args.abc_hidden_dim}, kl_beta={args.kl_beta}, "
          f"kl_warmup={args.kl_warmup_steps}, sigma_min={args.sigma_min}, "
          f"lr={args.abc_learning_rate}, epochs={args.num_epochs}, batch={args.batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}, g_init={args.g_init}, "
          f"max_len={args.max_length}")
    print("=" * 70)

    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    num_layers = model_wrapper.num_layers
    print(f"Model has {num_layers} layers")

    print("\nLoading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support: {len(support_samples)}, Test: {len(test_samples)}")

    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = list(range(0, num_layers, args.layer_step))

    print(f"\nLayers to test: {layers}")
    print(f"Total: {len(layers)} layers")

    baseline_accuracy = args.baseline_accuracy
    if not args.skip_baseline:
        print("\n" + "-" * 70)
        print("Running baseline evaluation...")
        baseline = run_baseline_evaluation(
            model_wrapper,
            tokenizer,
            test_samples,
            args.dataset,
            args.max_new_tokens,
            args.num_beams,
            args.use_early_stopping,
            args.max_length,
        )
        baseline_accuracy = baseline["accuracy"]
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    elif baseline_accuracy is not None:
        print(f"\nUsing provided baseline accuracy: {baseline_accuracy:.2f}%")
    else:
        print("\nSkipping baseline evaluation (no baseline_accuracy provided)")
        baseline_accuracy = 0.0

    print("\n" + "-" * 70)
    print(f"Testing {len(layers)} layers with method: abc")
    print("-" * 70)

    results = []
    checkpoints = {}

    for layer_idx in tqdm(layers, desc="Layer sweep", ncols=100):
        print(f"\n>>> Layer {layer_idx}")
        try:
            abc_method = ABCCoTVector(
                model_wrapper=model_wrapper,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                dataset_type=args.dataset,
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

            if args.load_vectors_dir:
                checkpoint_path = os.path.join(args.load_vectors_dir, f"abc_L{layer_idx}.pt")
                if os.path.exists(checkpoint_path):
                    print(f"  Loading ABC checkpoint from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    abc_method.load_state_dict(checkpoint, device=model_wrapper.device)
                else:
                    print("  Checkpoint not found, training new ABC model...")
                    abc_method.train(support_samples)
            else:
                abc_method.train(support_samples)

            if args.save_vector:
                checkpoint_path = os.path.join(output_dir, f"abc_L{layer_idx}.pt")
                torch.save({**abc_method.get_state_dict(), "args": vars(args)}, checkpoint_path)
                checkpoints[layer_idx] = checkpoint_path
                print(f"  Saved ABC checkpoint to {checkpoint_path}")

            eval_result = abc_method.eval(
                test_samples=test_samples,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                use_early_stopping=args.use_early_stopping,
                max_length=args.max_length,
            )

            diff = eval_result["accuracy"] - baseline_accuracy if baseline_accuracy else 0
            gate_value = abc_method.gate.item()
            entry = {
                "layer": layer_idx,
                "accuracy": eval_result["accuracy"],
                "diff": diff,
                "correct": eval_result["correct"],
                "total": eval_result["total"],
                "gate": gate_value,
            }
            results.append(entry)

            print(f"  Layer {layer_idx:2d}: {eval_result['accuracy']:.2f}% "
                  f"({eval_result['correct']}/{eval_result['total']}) "
                  f"[{diff:+.2f}% vs baseline] gate={gate_value:.3f}")

        except RuntimeError as error:
            if "out of memory" in str(error).lower() or "cuda" in str(error).lower():
                print(f"  Layer {layer_idx}: CUDA OOM - {error}")
                torch.cuda.empty_cache()
                results.append({
                    "layer": layer_idx,
                    "accuracy": 0,
                    "diff": -baseline_accuracy if baseline_accuracy else 0,
                    "error": f"CUDA OOM: {str(error)[:100]}",
                })
                continue

            print(f"  Layer {layer_idx}: Error - {error}")
            results.append({
                "layer": layer_idx,
                "accuracy": 0,
                "diff": -baseline_accuracy if baseline_accuracy else 0,
                "error": str(error),
            })
            continue

        except Exception as error:  # noqa: BLE001
            print(f"  Layer {layer_idx}: Error - {error}")
            results.append({
                "layer": layer_idx,
                "accuracy": 0,
                "diff": -baseline_accuracy if baseline_accuracy else 0,
                "error": str(error),
            })
            continue

        torch.cuda.empty_cache()

    if not results:
        print("\nNo successful layer evaluations.")
        return

    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["accuracy"])
        print("\n" + "=" * 70)
        print("Best layer")
        print("=" * 70)
        print(f"Layer {best['layer']}: {best['accuracy']:.2f}% ({best['diff']:+.2f}% vs baseline)")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "method": "abc",
        "baseline_accuracy": baseline_accuracy,
        "results": sorted(results, key=lambda r: r["layer"]),
        "checkpoints": checkpoints,
        "args": vars(args),
    }
    summary_path = os.path.join(output_dir, f"layer_sweep_abc_{args.dataset}.json")
    torch.save(report, summary_path)
    print(f"\nSaved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
