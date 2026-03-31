import argparse
import json
import os

import torch
from transformers import GenerationConfig

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run interactive inference with an ablated refusal direction.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model.")
    parser.add_argument("--artifact_subdir", type=str, default="inference_ablation")
    parser.add_argument("--direction_path", type=str, default=None)
    parser.add_argument("--direction_metadata_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def get_default_paths(model_path: str, artifact_subdir: str):
    model_alias = os.path.basename(model_path.rstrip("/"))
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    artifact_dir = os.path.join(repo_root, "pipeline", "runs", model_alias, artifact_subdir)
    return (
        artifact_dir,
        os.path.join(artifact_dir, "direction.pt"),
        os.path.join(artifact_dir, "direction_metadata.json"),
    )


def load_direction_artifacts(model_base, direction_path, metadata_path):
    direction = torch.load(direction_path, map_location=model_base.model.device)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return direction, metadata


def generate_one(model_base, prompt, fwd_pre_hooks, fwd_hooks, max_new_tokens, temperature):
    inputs = model_base.tokenize_instructions_fn(instructions=[prompt])
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": model_base.tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    generation_config = GenerationConfig(**generation_kwargs)

    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        output = model_base.model.generate(
            input_ids=inputs.input_ids.to(model_base.model.device),
            attention_mask=inputs.attention_mask.to(model_base.model.device),
            generation_config=generation_config,
        )

    generated = output[0, inputs.input_ids.shape[-1]:]
    return model_base.tokenizer.decode(generated, skip_special_tokens=True).strip()


def interactive_loop(model_base, fwd_pre_hooks, fwd_hooks, max_new_tokens, temperature):
    while True:
        try:
            prompt = input("\nUser> ").strip()
        except EOFError:
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        response = generate_one(
            model_base=model_base,
            prompt=prompt,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"\nAssistant> {response}")


def main():
    args = parse_arguments()
    if args.prompt is None and not args.interactive:
        raise ValueError("Pass either --prompt or --interactive.")

    artifact_dir, default_direction_path, default_metadata_path = get_default_paths(args.model_path, args.artifact_subdir)
    direction_path = args.direction_path or default_direction_path
    metadata_path = args.direction_metadata_path or default_metadata_path

    model_base = construct_model_base(args.model_path)
    direction, metadata = load_direction_artifacts(model_base, direction_path, metadata_path)
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)

    print(f"artifact_dir={artifact_dir}")
    print(f"position={metadata['position']} layer={metadata['layer']} mode=ablation")

    if args.prompt is not None:
        response = generate_one(
            model_base=model_base,
            prompt=args.prompt,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(response)

    if args.interactive:
        interactive_loop(
            model_base=model_base,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
