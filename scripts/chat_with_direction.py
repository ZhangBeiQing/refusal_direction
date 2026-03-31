import argparse
import json
import os
import sys

import torch

from transformers import GenerationConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a model using a saved refusal direction.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the target model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "ablation", "actadd"],
        default="ablation",
        help="Intervention mode. `ablation` removes the refusal direction.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--direction_path",
        type=str,
        default=None,
        help="Path to direction.pt. Defaults to pipeline/runs/<model_alias>/direction.pt.",
    )
    parser.add_argument(
        "--direction_metadata_path",
        type=str,
        default=None,
        help="Path to direction_metadata.json. Defaults to pipeline/runs/<model_alias>/direction_metadata.json.",
    )
    return parser.parse_args()


def get_default_artifact_paths(model_path: str):
    model_alias = os.path.basename(model_path.rstrip("/"))
    artifact_dir = os.path.join("pipeline", "runs", model_alias)
    return (
        os.path.join(artifact_dir, "direction.pt"),
        os.path.join(artifact_dir, "direction_metadata.json"),
    )


def load_direction_artifacts(model_base, direction_path: str, metadata_path: str):
    direction = torch.load(direction_path, map_location=model_base.model.device)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return direction, metadata


def build_hooks(model_base, direction, metadata, mode: str):
    if mode == "baseline":
        return [], []
    if mode == "ablation":
        return get_all_direction_ablation_hooks(model_base, direction)
    if mode == "actadd":
        layer = metadata["layer"]
        fwd_pre_hooks = [
            (model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))
        ]
        return fwd_pre_hooks, []
    raise ValueError(f"Unknown mode: {mode}")


def generate_one(model_base, prompt: str, fwd_pre_hooks, fwd_hooks, max_new_tokens: int, temperature: float) -> str:
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


def interactive_loop(model_base, fwd_pre_hooks, fwd_hooks, max_new_tokens: int, temperature: float):
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
    args = parse_args()
    if args.prompt is None and not args.interactive:
        raise ValueError("Pass either --prompt or --interactive.")

    default_direction_path, default_metadata_path = get_default_artifact_paths(args.model_path)
    direction_path = args.direction_path or default_direction_path
    metadata_path = args.direction_metadata_path or default_metadata_path

    model_base = construct_model_base(args.model_path)
    direction, metadata = load_direction_artifacts(model_base, direction_path, metadata_path)
    fwd_pre_hooks, fwd_hooks = build_hooks(model_base, direction, metadata, args.mode)

    print(f"mode={args.mode} layer={metadata['layer']} pos={metadata['pos']}")
    print(f"direction_path={direction_path}")

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
