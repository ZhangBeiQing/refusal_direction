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
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--history_path", type=str, default=None, help="Load/save interactive chat history as JSON.")
    parser.add_argument("--no_history", action="store_true", help="Do not include previous turns as context.")
    parser.add_argument("--clear_history", action="store_true", help="Clear history_path before starting interactive chat.")
    parser.add_argument("--debug_hooks", action="store_true", help="Print hook registration and call counts.")
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


def wrap_hooks_for_debug(fwd_pre_hooks, fwd_hooks):
    counters = {"forward_pre": 0, "forward": 0}

    def wrap_pre(hook):
        def wrapped(module, input):
            counters["forward_pre"] += 1
            return hook(module, input)

        return wrapped

    def wrap_forward(hook):
        def wrapped(module, input, output):
            counters["forward"] += 1
            return hook(module, input, output)

        return wrapped

    wrapped_pre_hooks = [(module, wrap_pre(hook)) for module, hook in fwd_pre_hooks]
    wrapped_hooks = [(module, wrap_forward(hook)) for module, hook in fwd_hooks]
    return wrapped_pre_hooks, wrapped_hooks, counters


def load_history(path: str | None):
    if path is None or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError(f"History file must contain a JSON list: {path}")
    return history


def save_history(path: str | None, history):
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def build_instruction_with_history(history, prompt: str) -> str:
    if not history:
        return prompt

    lines = ["Conversation history:"]
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    lines.append(f"User: {prompt}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def generate_one(
    model_base,
    prompt: str,
    fwd_pre_hooks,
    fwd_hooks,
    max_new_tokens: int,
    temperature: float,
    history=None,
    debug_hook_counters=None,
) -> str:
    instruction = build_instruction_with_history(history or [], prompt)
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
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
    if debug_hook_counters is not None:
        print(
            "\nDebug hooks: "
            f"forward_pre_calls={debug_hook_counters['forward_pre']} "
            f"forward_calls={debug_hook_counters['forward']}"
        )
    return model_base.tokenizer.decode(generated, skip_special_tokens=True).strip()


def interactive_loop(
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    max_new_tokens: int,
    temperature: float,
    history_path,
    no_history,
    clear_history,
    debug_hook_counters,
):
    history = load_history(history_path)
    if clear_history:
        history.clear()
        save_history(history_path, history)
        print("History cleared at startup.")
    elif history_path and history and not no_history:
        print(f"Loaded {len(history)} history messages from {history_path}. Use --clear_history or :reset to start fresh.")
    print("Commands: exit/quit/:q to leave, :reset to clear history, :history to show history.")

    while True:
        try:
            prompt = input("\nUser> ").strip()
        except EOFError:
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", ":q"}:
            break
        if prompt == ":reset":
            history.clear()
            save_history(history_path, history)
            print("History cleared.")
            continue
        if prompt == ":history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue

        response = generate_one(
            model_base=model_base,
            prompt=prompt,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            history=[] if no_history else history,
            debug_hook_counters=debug_hook_counters,
        )
        print(f"\nAssistant> {response}")
        if not no_history:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})
            save_history(history_path, history)


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
    debug_hook_counters = None
    if args.debug_hooks:
        fwd_pre_hooks, fwd_hooks, debug_hook_counters = wrap_hooks_for_debug(fwd_pre_hooks, fwd_hooks)

    print(f"mode={args.mode} layer={metadata['layer']} pos={metadata['pos']}")
    print(f"direction_path={direction_path}")
    if args.debug_hooks:
        print(f"registered_forward_pre_hooks={len(fwd_pre_hooks)} registered_forward_hooks={len(fwd_hooks)}")

    if args.prompt is not None:
        response = generate_one(
            model_base=model_base,
            prompt=args.prompt,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            history=[],
            debug_hook_counters=debug_hook_counters,
        )
        print(response)

    if args.interactive:
        interactive_loop(
            model_base=model_base,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            history_path=args.history_path,
            no_history=args.no_history,
            clear_history=args.clear_history,
            debug_hook_counters=debug_hook_counters,
        )


if __name__ == "__main__":
    main()
