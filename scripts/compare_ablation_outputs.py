import argparse
import json
import os

import torch
from transformers import GenerationConfig

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_ablation_inference import get_default_paths, load_direction_artifacts
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare baseline vs ablation outputs on a small prompt set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model.")
    parser.add_argument("--artifact_subdir", type=str, default="inference_ablation")
    parser.add_argument("--direction_path", type=str, default=None)
    parser.add_argument("--direction_metadata_path", type=str, default=None)
    parser.add_argument("--dataset", choices=["harmful_train", "harmful_val"], default="harmful_val")
    parser.add_argument("--start", type=int, default=0, help="Start offset in the selected dataset split.")
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts to compare.")
    parser.add_argument("--prompt", action="append", default=None, help="Optional explicit prompt; can be passed multiple times.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to artifact_dir/compare_outputs_<dataset>_<start>_<limit>.json.",
    )
    return parser.parse_args()


def _load_prompts(repo_root: str, dataset_name: str, start: int, limit: int, prompt_overrides):
    if prompt_overrides:
        return prompt_overrides

    split_path = os.path.join(repo_root, "dataset", "splits", f"{dataset_name}.json")
    with open(split_path, "r") as f:
        payload = json.load(f)
    selected = payload[start:start + limit]
    return [row["instruction"] for row in selected]


def _build_generation_config(model_base, max_new_tokens: int, temperature: float):
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": model_base.tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    return GenerationConfig(**generation_kwargs)


def _generate_one(model_base, prompt: str, generation_config: GenerationConfig, fwd_pre_hooks=None, fwd_hooks=None):
    if fwd_pre_hooks is None:
        fwd_pre_hooks = []
    if fwd_hooks is None:
        fwd_hooks = []

    inputs = model_base.tokenize_instructions_fn(instructions=[prompt])
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        with torch.inference_mode():
            output = model_base.model.generate(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
                generation_config=generation_config,
            )

    generated = output[0, inputs.input_ids.shape[-1]:]
    return model_base.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_arguments()
    repo_root = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.dirname(repo_root)

    artifact_dir, default_direction_path, default_metadata_path = get_default_paths(args.model_path, args.artifact_subdir)
    direction_path = args.direction_path or default_direction_path
    metadata_path = args.direction_metadata_path or default_metadata_path

    prompts = _load_prompts(
        repo_root=repo_root,
        dataset_name=args.dataset,
        start=args.start,
        limit=args.limit,
        prompt_overrides=args.prompt,
    )

    model_base = construct_model_base(args.model_path)
    direction, metadata = load_direction_artifacts(model_base, direction_path, metadata_path)
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    generation_config = _build_generation_config(model_base, args.max_new_tokens, args.temperature)
    save_path = args.save_path
    if save_path is None:
        default_name = f"compare_outputs_{args.dataset}_{args.start}_{len(prompts)}.json"
        save_path = os.path.join(artifact_dir, default_name)

    print(f"artifact_dir={artifact_dir}")
    print(f"position={metadata['position']} layer={metadata['layer']} mode=ablation")
    print(f"dataset={args.dataset} start={args.start} limit={len(prompts)}")

    records = []

    for idx, prompt in enumerate(prompts, start=1):
        baseline = _generate_one(
            model_base=model_base,
            prompt=prompt,
            generation_config=generation_config,
        )
        ablation = _generate_one(
            model_base=model_base,
            prompt=prompt,
            generation_config=generation_config,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
        )

        print("\n" + "=" * 80)
        print(f"[{idx}] Prompt")
        print(prompt)
        print("\n[Baseline]")
        print(baseline)
        print("\n[Ablation]")
        print(ablation)

        records.append(
            {
                "index": idx,
                "prompt": prompt,
                "baseline": baseline,
                "ablation": ablation,
            }
        )

    payload = {
        "model_path": args.model_path,
        "artifact_dir": artifact_dir,
        "direction_path": direction_path,
        "direction_metadata_path": metadata_path,
        "direction_metadata": metadata,
        "dataset": args.dataset,
        "start": args.start,
        "limit": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "records": records,
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison archive to {save_path}")


if __name__ == "__main__":
    main()
