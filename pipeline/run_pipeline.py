import argparse
import hashlib
import json
import os
import random

from dataset.load_dataset import load_dataset, load_dataset_split

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.refusal_calibration import (
    cache_refusal_calibration_responses,
    derive_filtered_splits_and_refusal_toks,
    get_refusal_calibration_paths,
    load_judged_refusal_cache,
    run_refusal_judge_subprocess,
)
from pipeline.submodules.select_direction import get_refusal_scores, select_direction


def _stable_digest(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def _manifest_matches(manifest_path, expected_payload):
    return _load_manifest(manifest_path) == expected_payload


def _write_manifest(manifest_path, payload):
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _instruction_list_signature(instructions):
    return _stable_digest(instructions)


def _dataset_signature(dataset):
    normalized = [
        {
            "instruction": row["instruction"],
            "category": row.get("category"),
        }
        for row in dataset
    ]
    return _stable_digest(normalized)


def _get_direction_signature(cfg, intervention_label):
    if intervention_label == "baseline":
        return {"intervention_label": intervention_label}

    direction_path = os.path.join(cfg.artifact_path(), "direction.pt")
    direction_metadata_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    with open(direction_metadata_path, "r") as f:
        direction_metadata = json.load(f)

    return {
        "intervention_label": intervention_label,
        "direction_file_digest": _file_digest(direction_path),
        "direction_metadata": direction_metadata,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the refusal direction pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model")
    parser.add_argument("--refusal_judge_model_path", type=str, default=None, help="Path to the Nemotron refusal judge model")
    parser.add_argument("--refusal_judge_backend", type=str, choices=["vllm", "transformers"], default=None)
    parser.add_argument("--refusal_judge_gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--ce_loss_n_batches", type=int, default=None)
    parser.add_argument("--ce_loss_batch_size", type=int, default=None)
    parser.add_argument("--activation_batch_size", type=int, default=None)
    parser.add_argument("--completion_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_max_new_tokens", type=int, default=None)
    parser.add_argument("--disable_refusal_calibration_cache", action="store_true")
    parser.add_argument("--disable_artifact_cache", action="store_true")
    return parser.parse_args()


def build_config_from_args(args):
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    override_fields = [
        "n_train",
        "n_val",
        "n_test",
        "max_new_tokens",
        "ce_loss_n_batches",
        "ce_loss_batch_size",
        "activation_batch_size",
        "completion_batch_size",
        "refusal_calibration_batch_size",
        "refusal_calibration_max_new_tokens",
        "refusal_judge_gpu_memory_utilization",
    ]
    for field_name in override_fields:
        value = getattr(args, field_name)
        if value is not None:
            setattr(cfg, field_name, value)

    cfg.refusal_judge_model_path = args.refusal_judge_model_path
    if args.refusal_judge_backend is not None:
        cfg.refusal_judge_backend = args.refusal_judge_backend
    if args.disable_refusal_calibration_cache:
        cfg.reuse_refusal_calibration_cache = False
    if args.disable_artifact_cache:
        cfg.reuse_artifacts = False
    if os.environ.get("TOGETHER_API_KEY") is None and "llamaguard2" in cfg.jailbreak_eval_methodologies:
        cfg.jailbreak_eval_methodologies = tuple(m for m in cfg.jailbreak_eval_methodologies if m != "llamaguard2")

    return cfg


def load_and_sample_datasets(cfg):
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype="harmful", split="train", instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype="harmless", split="train", instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype="harmful", split="val", instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype="harmless", split="val", instructions_only=True), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(
            model_base.model,
            harmful_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=cfg.activation_batch_size,
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model,
            harmless_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=cfg.activation_batch_size,
        )
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(
            model_base.model,
            harmful_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=cfg.activation_batch_size,
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model,
            harmless_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            batch_size=cfg.activation_batch_size,
        )
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def calibrate_refusal_proxy(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    split_to_instructions = {
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
    }
    calibration_paths = get_refusal_calibration_paths(cfg)
    os.makedirs(calibration_paths["artifact_dir"], exist_ok=True)

    response_cache_exists = os.path.exists(calibration_paths["response_cache_path"])
    judged_cache_exists = os.path.exists(calibration_paths["judged_cache_path"])
    response_cache_manifest = {
        "response_format_version": "force_english_all_models_v3",
        "model_path": cfg.model_path,
        "split_signatures": {
            split_name: _instruction_list_signature(instructions)
            for split_name, instructions in split_to_instructions.items()
        },
        "max_new_tokens": cfg.refusal_calibration_max_new_tokens,
    }

    need_response_cache = not (
        cfg.reuse_refusal_calibration_cache
        and response_cache_exists
        and _manifest_matches(calibration_paths["response_cache_manifest_path"], response_cache_manifest)
    )

    if need_response_cache:
        cache_refusal_calibration_responses(
            model_base=model_base,
            split_to_instructions=split_to_instructions,
            output_path=calibration_paths["response_cache_path"],
            batch_size=cfg.completion_batch_size,
            max_new_tokens=cfg.refusal_calibration_max_new_tokens,
        )
        _write_manifest(calibration_paths["response_cache_manifest_path"], response_cache_manifest)
        response_cache_exists = True

    judged_cache_manifest = {
        "judge_model_path": cfg.refusal_judge_model_path,
        "judge_backend": cfg.refusal_judge_backend,
        "response_cache_file_digest": _file_digest(calibration_paths["response_cache_path"]) if response_cache_exists else None,
    }

    need_judged_cache = not (
        cfg.reuse_refusal_calibration_cache
        and judged_cache_exists
        and _manifest_matches(calibration_paths["judged_cache_manifest_path"], judged_cache_manifest)
    )

    if need_judged_cache:
        model_base.del_model()
        run_refusal_judge_subprocess(
            cfg=cfg,
            input_path=calibration_paths["response_cache_path"],
            output_path=calibration_paths["judged_cache_path"],
        )
        _write_manifest(calibration_paths["judged_cache_manifest_path"], judged_cache_manifest)
        model_base = construct_model_base(cfg.model_path)

    judged_payload = load_judged_refusal_cache(calibration_paths["judged_cache_path"])
    split_to_filtered_instructions, refusal_toks, summary = derive_filtered_splits_and_refusal_toks(
        judged_payload=judged_payload,
        tokenizer=model_base.tokenizer,
        cfg=cfg,
        fallback_refusal_toks=model_base.refusal_toks,
    )

    with open(calibration_paths["summary_path"], "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    model_base.refusal_toks = refusal_toks

    return (
        split_to_filtered_instructions.get("harmful_train", harmful_train),
        split_to_filtered_instructions.get("harmless_train", harmless_train),
        split_to_filtered_instructions.get("harmful_val", harmful_val),
        split_to_filtered_instructions.get("harmless_val", harmless_val),
        model_base,
    )


def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    import torch

    artifact_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(artifact_dir, exist_ok=True)
    mean_diffs_path = os.path.join(artifact_dir, "mean_diffs.pt")
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    manifest = {
        "model_path": cfg.model_path,
        "harmful_train_signature": _instruction_list_signature(harmful_train),
        "harmless_train_signature": _instruction_list_signature(harmless_train),
    }

    if cfg.reuse_artifacts and os.path.exists(mean_diffs_path) and _manifest_matches(manifest_path, manifest):
        print(f"Reusing cached candidate directions from {mean_diffs_path}")
        return torch.load(mean_diffs_path, map_location=model_base.model.device)

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
        batch_size=cfg.activation_batch_size,
    )

    _write_manifest(manifest_path, manifest)
    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    import torch

    artifact_dir = os.path.join(cfg.artifact_path(), "select_direction")
    os.makedirs(artifact_dir, exist_ok=True)
    direction_path = os.path.join(cfg.artifact_path(), "direction.pt")
    direction_metadata_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    mean_diffs_path = os.path.join(cfg.artifact_path(), "generate_directions", "mean_diffs.pt")
    manifest = {
        "model_path": cfg.model_path,
        "harmful_val_signature": _instruction_list_signature(harmful_val),
        "harmless_val_signature": _instruction_list_signature(harmless_val),
        "candidate_directions_file_digest": _file_digest(mean_diffs_path) if os.path.exists(mean_diffs_path) else None,
    }

    if (
        cfg.reuse_artifacts
        and os.path.exists(direction_path)
        and os.path.exists(direction_metadata_path)
        and _manifest_matches(manifest_path, manifest)
    ):
        print(f"Reusing cached selected direction from {direction_path}")
        with open(direction_metadata_path, "r") as f:
            metadata = json.load(f)
        direction = torch.load(direction_path, map_location=model_base.model.device)
        return metadata["pos"], metadata["layer"], direction

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=artifact_dir,
        batch_size=cfg.activation_batch_size,
    )

    with open(f"{cfg.artifact_path()}/direction_metadata.json", "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)
    torch.save(direction, f"{cfg.artifact_path()}/direction.pt")
    _write_manifest(manifest_path, manifest)

    return pos, layer, direction


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    completions_dir = os.path.join(cfg.artifact_path(), "completions")
    os.makedirs(completions_dir, exist_ok=True)
    completions_path = os.path.join(completions_dir, f"{dataset_name}_{intervention_label}_completions.json")
    manifest_path = os.path.join(completions_dir, f"{dataset_name}_{intervention_label}_completions_manifest.json")

    if dataset is None:
        dataset = load_dataset(dataset_name)

    manifest = {
        "model_path": cfg.model_path,
        "dataset_signature": _dataset_signature(dataset),
        "max_new_tokens": cfg.max_new_tokens,
        "direction_signature": _get_direction_signature(cfg, intervention_label),
    }

    if cfg.reuse_artifacts and os.path.exists(completions_path) and _manifest_matches(manifest_path, manifest):
        print(f"Reusing cached completions from {completions_path}")
        return

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=cfg.completion_batch_size,
        max_new_tokens=cfg.max_new_tokens,
    )
    with open(completions_path, "w") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)
    _write_manifest(manifest_path, manifest)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    evaluations_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json")
    completions_path = os.path.join(cfg.artifact_path(), f"completions/{dataset_name}_{intervention_label}_completions.json")
    manifest_path = os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations_manifest.json")
    manifest = {
        "completions_file_digest": _file_digest(completions_path) if os.path.exists(completions_path) else None,
        "methodologies": list(eval_methodologies),
    }

    if cfg.reuse_artifacts and os.path.exists(evaluations_path) and _manifest_matches(manifest_path, manifest):
        print(f"Reusing cached evaluation from {evaluations_path}")
        return

    with open(completions_path, "r") as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=list(eval_methodologies),
        evaluation_path=evaluations_path,
    )

    with open(evaluations_path, "w") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)
    _write_manifest(manifest_path, manifest)


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    loss_eval_dir = os.path.join(cfg.artifact_path(), "loss_evals")
    os.makedirs(loss_eval_dir, exist_ok=True)
    loss_eval_path = os.path.join(loss_eval_dir, f"{intervention_label}_loss_eval.json")
    manifest_path = os.path.join(loss_eval_dir, f"{intervention_label}_loss_eval_manifest.json")

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), "completions/harmless_baseline_completions.json")
    manifest = {
        "model_path": cfg.model_path,
        "intervention_label": intervention_label,
        "direction_signature": _get_direction_signature(cfg, intervention_label),
        "ce_loss_batch_size": cfg.ce_loss_batch_size,
        "ce_loss_n_batches": cfg.ce_loss_n_batches,
        "on_distribution_completions_digest": _file_digest(on_distribution_completions_file_path) if os.path.exists(on_distribution_completions_file_path) else None,
    }

    if cfg.reuse_artifacts and os.path.exists(loss_eval_path) and _manifest_matches(manifest_path, manifest):
        print(f"Reusing cached loss evaluation from {loss_eval_path}")
        return

    loss_evals = evaluate_loss(
        model_base,
        fwd_pre_hooks,
        fwd_hooks,
        batch_size=cfg.ce_loss_batch_size,
        n_batches=cfg.ce_loss_n_batches,
        completions_file_path=on_distribution_completions_file_path,
    )
    with open(loss_eval_path, "w") as f:
        json.dump(loss_evals, f, indent=4, ensure_ascii=False)
    _write_manifest(manifest_path, manifest)


def run_pipeline(cfg: Config):
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

    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    pos, layer, direction = select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions)

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [
        (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))
    ], []

    for dataset_name in cfg.evaluation_datasets:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, "baseline", dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, "ablation", dataset_name)
        generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, "actadd", dataset_name)

    for dataset_name in cfg.evaluation_datasets:
        evaluate_completions_and_save_results_for_dataset(cfg, "baseline", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, "ablation", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
        evaluate_completions_and_save_results_for_dataset(cfg, "actadd", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

    harmless_test = random.sample(load_dataset_split(harmtype="harmless", split="test"), cfg.n_test)

    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, "baseline", "harmless", dataset=harmless_test)
    actadd_refusal_pre_hooks, actadd_refusal_hooks = [
        (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=+1.0))
    ], []
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks, "actadd", "harmless", dataset=harmless_test)

    evaluate_completions_and_save_results_for_dataset(cfg, "baseline", "harmless", eval_methodologies=cfg.refusal_eval_methodologies)
    evaluate_completions_and_save_results_for_dataset(cfg, "actadd", "harmless", eval_methodologies=cfg.refusal_eval_methodologies)

    evaluate_loss_for_datasets(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, "baseline")
    evaluate_loss_for_datasets(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, "ablation")
    evaluate_loss_for_datasets(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, "actadd")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(build_config_from_args(args))
