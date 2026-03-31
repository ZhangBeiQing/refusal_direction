import argparse
import json
import os

import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import calibrate_refusal_proxy, filter_data, load_and_sample_datasets
from pipeline.submodules.generate_directions import get_mean_diff
from pipeline.submodules.select_direction import get_last_position_logits, get_refusal_scores, kl_div_fn
from pipeline.utils.hook_utils import (
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare an ablation-only inference direction.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model.")
    parser.add_argument("--refusal_judge_model_path", type=str, default=None, help="Optional refusal judge model path.")
    parser.add_argument("--refusal_judge_backend", type=str, choices=["vllm", "transformers"], default=None)
    parser.add_argument("--refusal_judge_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=4)
    parser.add_argument("--completion_batch_size", type=int, default=4)
    parser.add_argument("--refusal_calibration_batch_size", type=int, default=16)
    parser.add_argument("--refusal_calibration_max_new_tokens", type=int, default=96)
    parser.add_argument("--position", type=int, default=-1, help="Source position used to extract the direction.")
    parser.add_argument("--kl_threshold", type=float, default=0.1)
    parser.add_argument("--prune_layer_percentage", type=float, default=0.2)
    parser.add_argument("--artifact_subdir", type=str, default="inference_ablation")
    parser.add_argument("--disable_refusal_calibration_cache", action="store_true")
    return parser.parse_args()


def build_config_from_args(args):
    model_alias = os.path.basename(args.model_path.rstrip("/"))
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    cfg.n_train = args.n_train
    cfg.n_val = args.n_val
    cfg.activation_batch_size = args.activation_batch_size
    cfg.completion_batch_size = args.completion_batch_size
    cfg.refusal_calibration_batch_size = args.refusal_calibration_batch_size
    cfg.refusal_calibration_max_new_tokens = args.refusal_calibration_max_new_tokens
    cfg.refusal_judge_model_path = args.refusal_judge_model_path
    if args.refusal_judge_backend is not None:
        cfg.refusal_judge_backend = args.refusal_judge_backend
    if args.refusal_judge_gpu_memory_utilization is not None:
        cfg.refusal_judge_gpu_memory_utilization = args.refusal_judge_gpu_memory_utilization
    if args.disable_refusal_calibration_cache:
        cfg.reuse_refusal_calibration_cache = False
    return cfg


def get_inference_artifact_dir(cfg: Config, artifact_subdir: str):
    return os.path.join(cfg.artifact_path(), artifact_subdir)


def build_all_layer_ablation_hooks(model_base, direction):
    fwd_pre_hooks = [
        (model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction))
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    return fwd_pre_hooks, fwd_hooks


def select_best_ablation_direction(
    model_base,
    harmful_instructions,
    harmless_instructions,
    candidate_directions,
    position,
    batch_size,
    kl_threshold,
    prune_layer_percentage,
):
    n_layers = candidate_directions.shape[0]

    baseline_harmful_refusal = get_refusal_scores(
        model_base.model,
        harmful_instructions,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks,
        batch_size=batch_size,
    ).mean().item()

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        batch_size=batch_size,
    )

    kept_rows = []
    all_rows = []

    for layer in range(n_layers):
        direction = candidate_directions[layer]
        fwd_pre_hooks, fwd_hooks = build_all_layer_ablation_hooks(model_base, direction)

        harmful_refusal = get_refusal_scores(
            model_base.model,
            harmful_instructions,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size,
        ).mean().item()

        harmless_logits = get_last_position_logits(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=harmless_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size,
        )
        kl_div = kl_div_fn(baseline_harmless_logits, harmless_logits, mask=None).mean().item()

        row = {
            "position": position,
            "layer": layer,
            "refusal_score": harmful_refusal,
            "baseline_refusal_score": baseline_harmful_refusal,
            "kl_div_score": kl_div,
        }
        all_rows.append(row)

        if prune_layer_percentage is not None and layer >= int(n_layers * (1.0 - prune_layer_percentage)):
            continue
        if kl_threshold is not None and kl_div > kl_threshold:
            continue

        kept_rows.append(row)

    if not kept_rows:
        raise RuntimeError("No candidate inference directions survived filtering.")

    kept_rows = sorted(kept_rows, key=lambda row: (row["refusal_score"], row["kl_div_score"]))
    best = kept_rows[0]
    return best, all_rows, kept_rows


def main():
    args = parse_arguments()
    cfg = build_config_from_args(args)
    artifact_dir = get_inference_artifact_dir(cfg, args.artifact_subdir)
    os.makedirs(artifact_dir, exist_ok=True)

    model_base = construct_model_base(cfg.model_path)
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    if cfg.refusal_judge_model_path:
        harmful_train, harmless_train, harmful_val, harmless_val, model_base = calibrate_refusal_proxy(
            cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
        )
    else:
        harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
            cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
        )

    mean_diff = get_mean_diff(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        batch_size=cfg.activation_batch_size,
        positions=[args.position],
    )[0]

    best, all_rows, kept_rows = select_best_ablation_direction(
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        candidate_directions=mean_diff,
        position=args.position,
        batch_size=cfg.activation_batch_size,
        kl_threshold=args.kl_threshold,
        prune_layer_percentage=args.prune_layer_percentage,
    )

    best_direction = mean_diff[best["layer"]]
    torch.save(best_direction, os.path.join(artifact_dir, "direction.pt"))

    metadata = {
        "position": best["position"],
        "layer": best["layer"],
        "selection_method": "ablation_only",
        "baseline_refusal_score": best["baseline_refusal_score"],
        "refusal_score": best["refusal_score"],
        "kl_div_score": best["kl_div_score"],
        "kl_threshold": args.kl_threshold,
        "prune_layer_percentage": args.prune_layer_percentage,
    }
    with open(os.path.join(artifact_dir, "direction_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    with open(os.path.join(artifact_dir, "direction_evaluations.json"), "w") as f:
        json.dump(all_rows, f, indent=4, ensure_ascii=False)

    with open(os.path.join(artifact_dir, "direction_evaluations_filtered.json"), "w") as f:
        json.dump(kept_rows, f, indent=4, ensure_ascii=False)

    print(f"Saved inference direction to {artifact_dir}")
    print(
        f"Selected position={best['position']} layer={best['layer']} "
        f"refusal_score={best['refusal_score']:.4f} "
        f"baseline={best['baseline_refusal_score']:.4f} "
        f"kl={best['kl_div_score']:.4f}"
    )


if __name__ == "__main__":
    main()
