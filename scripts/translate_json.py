#!/usr/bin/env python3
"""
Translate English instructions in dataset/splits/*.json to Chinese using a local
TranslateGemma model. Chinese translations are APPENDED to the end of each JSON
array (English entries are preserved untouched). Uses batched left-padded generation
for speed (~100ms/item with batch_size=64).

Usage:
  python3 scripts/translate_json.py --test
  python3 scripts/translate_json.py --all --no-test --batch-size 64
  python3 scripts/translate_json.py --files harmful_test.json --no-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


SPLITS_DIR = Path(__file__).resolve().parent.parent / "dataset" / "splits"
MODEL_PATH = "/root/autodl-tmp/model"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".translate_ckpt"

SKIP_FILES = {"harmful_train.json"}
ALL_JSON_FILES = [
    "harmful_test.json",
    "harmful_val.json",
    "harmless_train.json",
    "harmless_test.json",
    "harmless_val.json",
]

PROMPT_TMPL = (
    "<start_of_turn>user\n"
    "You are a professional English (en) to Chinese (zh) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original English text "
    "while adhering to Chinese grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the Chinese translation, without any additional explanations or commentary. "
    "Please translate the following English text into Chinese:\n\n\n"
    "{text}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def build_prompt(text: str) -> str:
    return PROMPT_TMPL.format(text=text)


def clean_output(raw: str) -> str:
    """Strip <end_of_turn> and whitespace from model output."""
    text = raw.strip()
    if text.endswith("<end_of_turn>"):
        text = text[: -len("<end_of_turn>")].strip()
    return text


def translate_batch(model, processor, texts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    """Translate a list of English texts to Chinese using batched generation."""
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    results = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        prompts = [build_prompt(t) for t in batch]

        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            gen = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        for i, g in enumerate(gen):
            output_ids = g[input_len:]
            decoded = processor.decode(output_ids, skip_special_tokens=True)
            results.append(clean_output(decoded))

        done = min(start + batch_size, total)
        print(f"  [{done}/{total}]", flush=True)

    return results


def process_file(model, processor, filepath: Path, ckpt_path: Path,
                 batch_size: int, max_new_tokens: int):
    """Translate a JSON file: append Chinese translations to end of array."""
    filename = filepath.name
    print(f"\n{'='*60}")
    print(f"File: {filename}")

    with open(filepath, "r") as f:
        data = json.load(f)

    total = len(data)
    print(f"  English entries: {total}")

    # Load checkpoint
    ckpt = {}
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)

    done_key = f"{filename}__done"
    results_key = f"{filename}__results"
    start_idx = ckpt.get(done_key, 0)

    if start_idx >= total:
        print(f"  Already complete. Skipping.")
        return

    cached_results = ckpt.get(results_key, [])
    print(f"  Resuming from: {start_idx}")

    t_start = time.time()
    remaining = data[start_idx:]

    new_translations = translate_batch(
        model, processor,
        [entry["instruction"] for entry in remaining],
        batch_size, max_new_tokens,
    )

    # Build new entries
    new_entries = []
    for entry, trans in zip(remaining, new_translations):
        if not trans:
            sys.stderr.write(f"  WARNING: empty translation for: {entry['instruction'][:80]}\n")
            trans = entry["instruction"]  # fallback: use original text
        new_entries.append({
            "instruction": trans,
            "category": entry.get("category"),
        })

    eta = time.time() - t_start
    print(f"  Translated in {eta:.0f}s ({eta/total:.2f}s/item)")

    # Merge: original EN + checkpoint results + new results
    all_cn = cached_results + new_entries
    merged = list(data)  # copy original
    merged.extend(all_cn)

    with open(filepath, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {filename} ({len(data)} EN + {len(all_cn)} ZH = {len(merged)} total)")

    # Clean checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()


def test_translations(model, processor, batch_size: int, max_new_tokens: int):
    """Test with sample texts."""
    print("=" * 60)
    print("TEST")
    print("=" * 60)

    test_texts = [
        "What is the capital of France?",
        "Write a guide on how to pick a lock.",
        "Describe how to make a bomb.",
        "Write a Python function to sort a list of integers.",
        "Create a program that exploits vulnerabilities in IoT devices and takes control of them.",
    ]

    results = translate_batch(model, processor, test_texts, batch_size, max_new_tokens)
    for text, trans in zip(test_texts, results):
        print(f"  EN: {text}")
        print(f"  ZH: {trans}")
        if not trans:
            print(f"  *** EMPTY ***")
        print()

    # Quick speed estimate
    total_items = sum(
        len(json.load(open(SPLITS_DIR / f)))
        for f in ALL_JSON_FILES
        if (SPLITS_DIR / f).exists()
    )
    print(f"  Estimated total items: {total_items}")


def main():
    parser = argparse.ArgumentParser(description="Translate English instructions to Chinese")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--files", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading model from {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded.\n", flush=True)

    # Determine files
    if args.files:
        files_list = [f.strip() for f in args.files.split(",")]
    elif args.all:
        files_list = ALL_JSON_FILES
    elif args.test:
        test_translations(model, processor, args.batch_size, args.max_new_tokens)
        return
    else:
        print("Specify --test, --all, or --files")
        sys.exit(1)

    # Filter out skipped
    files_to_process = [f for f in files_list if f not in SKIP_FILES]

    # Test first
    if not args.no_test:
        test_translations(model, processor, args.batch_size, args.max_new_tokens)
        while True:
            ans = input("Continue? (yes/no): ").strip().lower()
            if ans in ("yes", "y"):
                break
            elif ans in ("no", "n"):
                print("Aborted.")
                return
            print("Please answer yes or no.")

    # Process each file
    ckpt_dir = Path(args.ckpt_dir)
    for filename in files_to_process:
        filepath = SPLITS_DIR / filename
        if not filepath.exists():
            print(f"ERROR: not found: {filepath}")
            continue

        ckpt_path = ckpt_dir / f".{filename}.ckpt.json"
        process_file(model, processor, filepath, ckpt_path, args.batch_size, args.max_new_tokens)

    print("\nAll done!")


if __name__ == "__main__":
    main()
