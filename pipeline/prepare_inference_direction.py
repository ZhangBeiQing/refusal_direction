import argparse
import json
import os

import torch

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import (
    _instruction_list_signature,
    _manifest_matches,
    _write_manifest,
    calibrate_refusal_proxy,
    filter_data,
    load_and_sample_datasets,
)
from pipeline.submodules.generate_directions import get_mean_diff
from pipeline.submodules.select_direction import get_refusal_scores, kl_div_fn
from pipeline.utils.hook_utils import (
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    add_hooks,
)
from pipeline.utils.logging import get_logger, enable_file_logging

logger = get_logger("PrepareInferenceDirection")

HARMLESS_KL_PROGRESS_LOG_INTERVAL_BATCHES = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prepare an ablation-only inference direction.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model.")
    parser.add_argument("--refusal_judge_model_path", type=str, default=None, help="Optional refusal judge model path.")
    parser.add_argument("--refusal_judge_backend", type=str, choices=["vllm", "transformers"], default=None)
    parser.add_argument("--refusal_judge_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--n_train", type=int, default=-1)
    parser.add_argument("--n_val", type=int, default=-1)
    parser.add_argument("--activation_batch_size", type=int, default=None)
    parser.add_argument("--completion_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_max_new_tokens", type=int, default=None)
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
    _set_if_not_none(cfg, "activation_batch_size", args.activation_batch_size)
    _set_if_not_none(cfg, "completion_batch_size", args.completion_batch_size)
    _set_if_not_none(cfg, "refusal_calibration_batch_size", args.refusal_calibration_batch_size)
    _set_if_not_none(cfg, "refusal_calibration_max_new_tokens", args.refusal_calibration_max_new_tokens)
    cfg.refusal_judge_model_path = args.refusal_judge_model_path
    if args.refusal_judge_backend is not None:
        cfg.refusal_judge_backend = args.refusal_judge_backend
    if args.refusal_judge_gpu_memory_utilization is not None:
        cfg.refusal_judge_gpu_memory_utilization = args.refusal_judge_gpu_memory_utilization
    if args.disable_refusal_calibration_cache:
        cfg.reuse_refusal_calibration_cache = False
    return cfg


def _set_if_not_none(obj, attr, value):
    if value is not None:
        setattr(obj, attr, value)


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


def _get_last_position_logits_for_batch(model_base, instructions, fwd_pre_hooks=None, fwd_hooks=None):
    if fwd_pre_hooks is None:
        fwd_pre_hooks = []
    if fwd_hooks is None:
        fwd_hooks = []

    tokenized = model_base.tokenize_instructions_fn(instructions=instructions)
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        with torch.inference_mode():
            logits = model_base.model(
                input_ids=tokenized.input_ids.to(model_base.model.device),
                attention_mask=tokenized.attention_mask.to(model_base.model.device),
            ).logits
    return logits[:, -1, :]


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

    logger.info("计算 baseline refusal score (harmful)...")
    baseline_harmful_refusal = get_refusal_scores(
        model_base.model,
        harmful_instructions,
        model_base.tokenize_instructions_fn,
        model_base.refusal_toks,
        batch_size=batch_size,
    ).mean().item()

    logger.info("按 batch 流式计算 harmless KL，避免缓存全量 logits...")

    harmful_refusal_by_layer = []
    layer_hooks = []

    for layer in range(n_layers):
        direction = candidate_directions[layer]
        logger.debug("计算 harmful refusal: layer=%d/%d ...", layer, n_layers)
        fwd_pre_hooks, fwd_hooks = build_all_layer_ablation_hooks(model_base, direction)
        layer_hooks.append((fwd_pre_hooks, fwd_hooks))
        harmful_refusal = get_refusal_scores(
            model_base.model,
            harmful_instructions,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size,
        ).mean().item()
        harmful_refusal_by_layer.append(harmful_refusal)

    kl_sums = [0.0 for _ in range(n_layers)]
    total_harmless_examples = 0
    total_harmless_batches = (len(harmless_instructions) + batch_size - 1) // batch_size

    for batch_idx, start_idx in enumerate(range(0, len(harmless_instructions), batch_size), start=1):
        batch_instructions = harmless_instructions[start_idx:start_idx + batch_size]
        _maybe_log_harmless_kl_progress(
            batch_idx=batch_idx,
            total_batches=total_harmless_batches,
            batch_start=start_idx,
            batch_size=batch_size,
            total_examples=len(harmless_instructions),
        )
        baseline_logits = _get_last_position_logits_for_batch(
            model_base=model_base,
            instructions=batch_instructions,
        )
        total_harmless_examples += baseline_logits.shape[0]

        for layer in range(n_layers):
            fwd_pre_hooks, fwd_hooks = layer_hooks[layer]
            intervention_logits = _get_last_position_logits_for_batch(
                model_base=model_base,
                instructions=batch_instructions,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
            )
            batch_kl = kl_div_fn(baseline_logits, intervention_logits, mask=None)
            kl_sums[layer] += batch_kl.sum().item()

    kept_rows = []
    all_rows = []

    for layer in range(n_layers):
        logger.debug("评估 layer=%d/%d ...", layer, n_layers)
        harmful_refusal = harmful_refusal_by_layer[layer]
        kl_div = kl_sums[layer] / max(total_harmless_examples, 1)

        row = {
            "position": position,
            "layer": layer,
            "refusal_score": harmful_refusal,
            "baseline_refusal_score": baseline_harmful_refusal,
            "kl_div_score": kl_div,
        }
        all_rows.append(row)

        if (prune_layer_percentage is not None and layer >= int(n_layers * (1.0 - prune_layer_percentage))) or \
           (kl_threshold is not None and kl_div > kl_threshold):
            logger.info("  layer=%2d  refusal=%.4f  kl=%.4f  [pruned]", layer, harmful_refusal, kl_div)
        else:
            logger.info("  layer=%2d  refusal=%.4f  kl=%.4f  [kept]", layer, harmful_refusal, kl_div)

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


def _format_count(items):
    if isinstance(items, int):
        return items
    return len(items)


def _load_or_compute_mean_diff(cfg, model_base, harmful_train, harmless_train, position, artifact_dir):
    mean_diff_path = os.path.join(artifact_dir, "mean_diff.pt")
    manifest_path = os.path.join(artifact_dir, "mean_diff_manifest.json")
    manifest = {
        "model_path": cfg.model_path,
        "position": position,
        "harmful_train_signature": _instruction_list_signature(harmful_train),
        "harmless_train_signature": _instruction_list_signature(harmless_train),
    }

    if os.path.exists(mean_diff_path) and _manifest_matches(manifest_path, manifest):
        logger.info("  复用已有 mean_diff 缓存: %s", mean_diff_path)
        return torch.load(mean_diff_path, map_location=model_base.model.device)

    mean_diff = get_mean_diff(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        batch_size=cfg.activation_batch_size,
        positions=[position],
    )[0]
    torch.save(mean_diff, mean_diff_path)
    _write_manifest(manifest_path, manifest)
    logger.info("  mean_diff 已写入: %s", mean_diff_path)
    return mean_diff


def _maybe_log_harmless_kl_progress(batch_idx, total_batches, batch_start, batch_size, total_examples):
    if batch_idx == 1 or batch_idx == total_batches or batch_idx % HARMLESS_KL_PROGRESS_LOG_INTERVAL_BATCHES == 0:
        batch_end = min(batch_start + batch_size, total_examples)
        logger.info(
            "  harmless KL 进度: batch %d/%d, examples [%d:%d)/%d",
            batch_idx,
            total_batches,
            batch_start,
            batch_end,
            total_examples,
        )


def main():
    args = parse_arguments()
    cfg = build_config_from_args(args)
    artifact_dir = get_inference_artifact_dir(cfg, args.artifact_subdir)
    os.makedirs(artifact_dir, exist_ok=True)
    enable_file_logging(os.path.join(artifact_dir, "logs"))

    logger.info("=" * 60)
    logger.info("[Stage 1/5] 加载模型...")
    model_base = construct_model_base(cfg.model_path)
    logger.info("  模型已加载: %s", cfg.model_path)
    logger.info("  layers=%d  hidden_size=%d",
                model_base.model.config.num_hidden_layers,
                model_base.model.config.hidden_size)
    logger.info("  默认 refusal_toks=%s", model_base.refusal_toks)

    logger.info("[Stage 2/5] 加载并采样数据集...")
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    logger.info("  harmful_train: %d 条", _format_count(harmful_train))
    logger.info("  harmless_train: %d 条", _format_count(harmless_train))
    logger.info("  harmful_val: %d 条", _format_count(harmful_val))
    logger.info("  harmless_val: %d 条", _format_count(harmless_val))

    if cfg.refusal_judge_model_path:
        logger.info("[Stage 3/5] Refusal 校准 (Nemotron judge)...")
        logger.info("  judge 模型: %s", cfg.refusal_judge_model_path)
        logger.info("  每个 split 的指令总数: train=%d  val=%d",
                    _format_count(harmful_train) + _format_count(harmless_train),
                    _format_count(harmful_val) + _format_count(harmless_val))
        harmful_train, harmless_train, harmful_val, harmless_val, model_base = calibrate_refusal_proxy(
            cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
        )
        logger.info("  校准后 refusal_toks=%s", model_base.refusal_toks)
        logger.info("  过滤后 harmful_train=%d  harmless_train=%d",
                    _format_count(harmful_train), _format_count(harmless_train))
        logger.info("  过滤后 harmful_val=%d  harmless_val=%d",
                    _format_count(harmful_val), _format_count(harmless_val))
    else:
        logger.info("[Stage 3/5] Refusal score 过滤...")
        harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
            cfg,
            model_base,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
        )
        logger.info("  过滤后 harmful_train=%d  harmless_train=%d",
                    _format_count(harmful_train), _format_count(harmless_train))
        logger.info("  过滤后 harmful_val=%d  harmless_val=%d",
                    _format_count(harmful_val), _format_count(harmless_val))

    logger.info("[Stage 4/5] 计算 mean difference direction...")
    logger.info("  position=%d  batch_size=%d", args.position, cfg.activation_batch_size)
    logger.info("  harmful 指令数=%d  harmless 指令数=%d",
                _format_count(harmful_train), _format_count(harmless_train))
    mean_diff = _load_or_compute_mean_diff(
        cfg=cfg,
        model_base=model_base,
        harmful_train=harmful_train,
        harmless_train=harmless_train,
        position=args.position,
        artifact_dir=artifact_dir,
    )
    logger.info("  mean_diff shape=%s", mean_diff.shape)

    logger.info("[Stage 5/5] 选择最优 ablation direction (共 %d 层)...", mean_diff.shape[0])
    logger.info("  kl_threshold=%.2f  prune_layer_percentage=%.2f",
                args.kl_threshold, args.prune_layer_percentage)
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
    logger.info("  survived=%d/%d layers", len(kept_rows), len(all_rows))
    for row in kept_rows[:5]:
        logger.info("    layer=%2d  refusal=%.4f  baseline=%.4f  kl=%.4f",
                    row["layer"], row["refusal_score"],
                    row["baseline_refusal_score"], row["kl_div_score"])

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

    print()
    print("=" * 60)
    print(f"Saved inference direction to {artifact_dir}")
    print(
        f"Selected position={best['position']} layer={best['layer']} "
        f"refusal_score={best['refusal_score']:.4f} "
        f"baseline={best['baseline_refusal_score']:.4f} "
        f"kl={best['kl_div_score']:.4f}"
    )


if __name__ == "__main__":
    main()
