import argparse
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the refusal direction pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model")
    parser.add_argument("--refusal_judge_model_path", type=str, default=None, help="Path to the Nemotron refusal judge model")
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
    ]
    for field_name in override_fields:
        value = getattr(args, field_name)
        if value is not None:
            setattr(cfg, field_name, value)

    cfg.refusal_judge_model_path = args.refusal_judge_model_path
    if args.disable_refusal_calibration_cache:
        cfg.reuse_refusal_calibration_cache = False
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

    need_response_cache = not (cfg.reuse_refusal_calibration_cache and response_cache_exists)
    need_judged_cache = not (cfg.reuse_refusal_calibration_cache and judged_cache_exists)

    if need_response_cache:
        cache_refusal_calibration_responses(
            model_base=model_base,
            split_to_instructions=split_to_instructions,
            output_path=calibration_paths["response_cache_path"],
            batch_size=cfg.completion_batch_size,
            max_new_tokens=cfg.refusal_calibration_max_new_tokens,
        )

    if need_judged_cache:
        model_base.del_model()
        run_refusal_judge_subprocess(
            cfg=cfg,
            input_path=calibration_paths["response_cache_path"],
            output_path=calibration_paths["judged_cache_path"],
        )
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
    artifact_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(artifact_dir, exist_ok=True)

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
        batch_size=cfg.activation_batch_size,
    )

    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    artifact_dir = os.path.join(cfg.artifact_path(), "select_direction")
    os.makedirs(artifact_dir, exist_ok=True)

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

    import torch

    torch.save(direction, f"{cfg.artifact_path()}/direction.pt")

    return pos, layer, direction


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    completions_dir = os.path.join(cfg.artifact_path(), "completions")
    os.makedirs(completions_dir, exist_ok=True)

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=cfg.completion_batch_size,
        max_new_tokens=cfg.max_new_tokens,
    )

    with open(f"{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json", "w") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    with open(os.path.join(cfg.artifact_path(), f"completions/{dataset_name}_{intervention_label}_completions.json"), "r") as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=list(eval_methodologies),
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f"{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json", "w") as f:
        json.dump(evaluation, f, indent=4, ensure_ascii=False)


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    loss_eval_dir = os.path.join(cfg.artifact_path(), "loss_evals")
    os.makedirs(loss_eval_dir, exist_ok=True)

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), "completions/harmless_baseline_completions.json")

    loss_evals = evaluate_loss(
        model_base,
        fwd_pre_hooks,
        fwd_hooks,
        batch_size=cfg.ce_loss_batch_size,
        n_batches=cfg.ce_loss_n_batches,
        completions_file_path=on_distribution_completions_file_path,
    )

    with open(f"{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json", "w") as f:
        json.dump(loss_evals, f, indent=4, ensure_ascii=False)


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
