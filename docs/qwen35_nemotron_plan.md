# Qwen3.5 + Nemotron Adaptation Plan

## Goal

Adapt this project to run the refusal-direction pipeline with:

- target model: `Qwen3.5-4B`
- judge model: `nemotron-content-safety-reasoning-4b`

while preserving the original `generate_directions -> select_direction -> intervention` workflow as much as possible.

## Main Problems

### 1. The current refusal proxy is too model-specific

The current pipeline uses `self.refusal_toks` as a fast proxy for refusal. In the existing Qwen implementation this is derived from English prefixes such as:

- `I`
- `As`

This is likely inaccurate for `Qwen3.5-4B` because:

- the dataset now contains Chinese harmful prompts
- Qwen3.5 may refuse in Chinese
- Qwen3.5 defaults to thinking mode, which changes the first generated tokens

### 2. We only have one 16GB GPU

We cannot keep both:

- `Qwen3.5-4B`
- `nemotron-content-safety-reasoning-4b`

running at the same time under vLLM, so the workflow must be staged.

## Agreed Direction

We will not replace the whole refusal-direction algorithm with Nemotron-based judging.

Instead, we will:

1. use `Qwen3.5-4B` to generate responses for the train/val split instructions
2. use `nemotron-content-safety-reasoning-4b` to judge whether those responses are refusals
3. use those refusal labels to:
   - rebuild the filtered harmful/harmless train/val sets
   - rebuild `refusal_toks`
4. keep the rest of the pipeline unchanged:
   - `generate_directions`
   - `select_direction`
   - direction ablation
   - activation addition

This keeps the expensive judge model out of the inner candidate-direction search loop.

## Why Not Replace `select_direction` With Nemotron Directly

`select_direction` evaluates many candidate directions. Its current scoring loop depends on a fast refusal proxy computed from last-position logits.

If we replaced this with:

- generate a full response
- shut down Qwen
- run Nemotron
- score refusal

for every candidate direction, the cost would become too high.

So Nemotron should be used to calibrate the proxy, not to replace the entire search loop.

## Required Changes

### A. Force Qwen3.5 into non-thinking mode

The Qwen3.5 chat template currently defaults to thinking mode. This must be disabled globally when preparing instruction prompts for:

- scoring
- generation
- activation extraction

Otherwise the first generated tokens may be `<think>`-related tokens instead of normal answer-start tokens, which breaks refusal-token estimation.

Implementation direction:

- update the Qwen tokenizer/chat-template path to pass `enable_thinking=False`

### B. Stage-1 cache: generate base-model responses

For the following splits:

- `harmful_train`
- `harmful_val`
- `harmless_train`
- `harmless_val`

run `Qwen3.5-4B` once and save, for each example:

- `instruction`
- `category`
- `response`
- ideally `first_response_token_id`

This cache should be written to disk so Qwen does not need to be re-run when Nemotron is started later.

### Why include harmless splits

We agreed not to judge only `harmful_xxx.json`.

Reason:

- `filter_data(...)` currently filters both harmful and harmless sets
- `generate_directions(...)` uses harmful vs harmless mean activations
- if harmless splits contain many refusal-like responses, the mean-difference direction becomes noisy
- `select_direction(...)` also uses harmless instructions for steering evaluation

Therefore Nemotron-based refusal labels should be computed for:

- harmful train/val
- harmless train/val

### C. Stage-2 judge: use Nemotron to label refusal

After response caches are produced, unload/stop Qwen and run:

- `nemotron-content-safety-reasoning-4b`

on the cached `(instruction, response)` pairs.

The desired output is not a generic harmfulness label alone. We specifically need a refusal-style label such as:

- `is_refusal = 1`
- `is_refusal = 0`

These labels will become the new source of truth for calibration.

### D. Rebuild filtered datasets

Use the Nemotron refusal labels to rebuild the train/val filtering behavior.

Target behavior:

- harmful split: keep examples where the base model refuses
- harmless split: keep examples where the base model does not refuse

This replaces the current `filter_data(...)` logic that depends on `get_refusal_scores(...) > 0` or `< 0`.

### E. Rebuild `refusal_toks`

Use the judged harmful-refusal examples to derive a new token-level refusal proxy.

Suggested procedure:

1. collect examples judged as refusal by Nemotron
2. look at their `first_response_token_id`
3. count token frequency
4. choose a small set of high-frequency start tokens as the new `refusal_toks`

This gives us a Qwen3.5-specific, data-calibrated refusal-token set instead of hardcoded English-only prefixes.

## What Stays the Same

After the new filtered sets and new `refusal_toks` are built, keep the original pipeline logic:

- `generate_directions(...)`
- `select_direction(...)`
- ablation hooks
- activation addition hooks
- final completion generation

This means the refusal-direction method itself is preserved. Only the calibration and filtering layers are replaced.

## Important Clarification

The reason to rebuild `refusal_toks` is not mainly about choosing `position`.

The real reason is that `select_direction(...)` still uses `get_refusal_scores(...)` as its fast objective:

- baseline refusal score
- ablation refusal score
- steering refusal score

If `refusal_toks` remains wrong, then `select_direction(...)` will optimize the wrong target even if dataset filtering is fixed.

## Open Technical Checks

These should be verified when implementing:

1. `Qwen3.5-4B` module paths in `qwen_model.py`
   - block modules
   - attention modules
   - MLP modules

2. prompt formatting path for Qwen3.5
   - use chat template
   - disable thinking

3. whether `first_response_token_id` should be stored during generation directly
   - preferred, to avoid re-tokenizing decoded text later

4. exact Nemotron prompt format for refusal judging
   - must distinguish refusal from merely safe-but-nonrefusal answers

## Summary

Final agreed plan:

- Qwen3.5 generates train/val responses in non-thinking mode
- Nemotron judges refusal on cached responses
- those labels rebuild:
  - filtered harmful/harmless train/val sets
  - `refusal_toks`
- the rest of the refusal-direction pipeline stays unchanged

This is the minimum-change path that keeps the original paper/code structure but removes the fragile `I/As` refusal-token assumption.
