import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)
from pipeline.utils.wandb_utils import wandb_artifact, wandb_log


@dataclass
class GeometryOptimizationConfig:
    method: str
    cone_dim: int
    epochs: int
    batch_size: int
    effective_batch_size: int
    learning_rate: float
    target_max_new_tokens: int
    n_cone_samples: int
    ablation_lambda: float
    addition_lambda: float
    retain_lambda: float
    init_from_dim: bool


def _stable_digest(value) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _tensor_digest(tensor: Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def _load_manifest(manifest_path: str):
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def _manifest_matches(manifest_path: str, expected_payload) -> bool:
    return _load_manifest(manifest_path) == expected_payload


def _write_json(path: str, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _write_manifest(manifest_path: str, payload):
    _write_json(manifest_path, payload)


def _instruction_list_signature(instructions: List[str]) -> str:
    return _stable_digest(instructions)


def _balanced_instruction_pairs(harmful_instructions: List[str], harmless_instructions: List[str]):
    n_examples = min(len(harmful_instructions), len(harmless_instructions))
    return harmful_instructions[:n_examples], harmless_instructions[:n_examples]


def _records_for_generation(instructions: List[str], category: str):
    return [{"instruction": instruction, "category": category} for instruction in instructions]


def generate_geometry_targets(
    cfg,
    model_base: ModelBase,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
    base_direction: Float[Tensor, "d_model"],
    add_layer: int,
    artifact_dir: str,
):
    os.makedirs(artifact_dir, exist_ok=True)
    targets_path = os.path.join(artifact_dir, "targets.json")
    manifest_path = os.path.join(artifact_dir, "targets_manifest.json")

    harmful_instructions, harmless_instructions = _balanced_instruction_pairs(harmful_instructions, harmless_instructions)
    manifest = {
        "model_path": cfg.model_path,
        "harmful_train_signature": _instruction_list_signature(harmful_instructions),
        "harmless_train_signature": _instruction_list_signature(harmless_instructions),
        "base_direction_digest": _tensor_digest(base_direction),
        "add_layer": add_layer,
        "target_max_new_tokens": cfg.rdo_target_max_new_tokens,
    }

    if cfg.reuse_artifacts and os.path.exists(targets_path) and _manifest_matches(manifest_path, manifest):
        with open(targets_path, "r") as f:
            return json.load(f)

    harmful_dataset = _records_for_generation(harmful_instructions, "harmful_train")
    harmless_dataset = _records_for_generation(harmless_instructions, "harmless_train")

    ablation_pre_hooks, ablation_hooks = get_all_direction_ablation_hooks(model_base, base_direction)
    harmful_ablation = model_base.generate_completions(
        harmful_dataset,
        fwd_pre_hooks=ablation_pre_hooks,
        fwd_hooks=ablation_hooks,
        batch_size=cfg.completion_batch_size,
        max_new_tokens=cfg.rdo_target_max_new_tokens,
    )

    addition_pre_hooks = [
        (
            model_base.model_block_modules[add_layer],
            get_activation_addition_input_pre_hook(vector=base_direction, coeff=+1.0),
        )
    ]
    harmless_addition = model_base.generate_completions(
        harmless_dataset,
        fwd_pre_hooks=addition_pre_hooks,
        fwd_hooks=[],
        batch_size=cfg.completion_batch_size,
        max_new_tokens=cfg.rdo_target_max_new_tokens,
    )
    harmless_retain = model_base.generate_completions(
        harmless_dataset,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=cfg.completion_batch_size,
        max_new_tokens=max(1, cfg.rdo_target_max_new_tokens - 1),
    )

    targets = {
        "harmful": [
            {
                "instruction": record["prompt"],
                "ablation_target": record["response"],
            }
            for record in harmful_ablation
        ],
        "harmless": [
            {
                "instruction": add_record["prompt"],
                "addition_target": add_record["response"],
                "retain_target": retain_record["response"],
            }
            for add_record, retain_record in zip(harmless_addition, harmless_retain)
        ],
    }

    _write_json(targets_path, targets)
    _write_manifest(manifest_path, manifest)
    return targets


def _build_training_records(targets: Dict[str, List[Dict[str, str]]]):
    records = []
    for harmful_record, harmless_record in zip(targets["harmful"], targets["harmless"]):
        if not harmful_record["ablation_target"].strip():
            continue
        if not harmless_record["addition_target"].strip():
            continue
        if not harmless_record["retain_target"].strip():
            continue
        records.append(
            {
                "harmful_instruction": harmful_record["instruction"],
                "harmless_instruction": harmless_record["instruction"],
                "ablation_target": harmful_record["ablation_target"],
                "addition_target": harmless_record["addition_target"],
                "retain_target": harmless_record["retain_target"],
            }
        )

    if not records:
        raise ValueError("No non-empty RDO training targets were generated.")

    return records


def _find_prompt_end_mask(input_ids: Tensor, attention_mask: Tensor, eoi_toks: Tensor, prompt_lengths: List[int]):
    loss_mask = attention_mask.clone()
    loss_mask[:, -1] = 0

    for batch_idx in range(input_ids.shape[0]):
        found = False
        for token_idx in range(input_ids.shape[1]):
            token_window = input_ids[batch_idx, token_idx:token_idx + eoi_toks.shape[0]]
            if token_window.shape[0] != eoi_toks.shape[0]:
                continue
            if torch.all(token_window.cpu() == eoi_toks):
                loss_mask[batch_idx, : token_idx + eoi_toks.shape[0] - 1] = 0
                found = True
                break
            if eoi_toks.shape[0] == 6 and (token_window.cpu() == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                loss_mask[batch_idx, : token_idx + eoi_toks.shape[0] - 1] = 0
                found = True
                break
        if not found:
            sequence_tokens = int(attention_mask[batch_idx].sum().item())
            pad_tokens = input_ids.shape[1] - sequence_tokens
            loss_mask[batch_idx, : pad_tokens + prompt_lengths[batch_idx]] = 0

    return loss_mask


def tokenize_targets(model_base: ModelBase, instructions: List[str], targets: List[str]):
    inputs = model_base.tokenize_instructions_fn(instructions=instructions, outputs=targets)
    prompt_inputs = model_base.tokenize_instructions_fn(instructions=instructions)
    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
    eoi_toks = torch.tensor(model_base.eoi_toks)
    loss_mask = _find_prompt_end_mask(inputs["input_ids"], inputs["attention_mask"], eoi_toks, prompt_lengths)
    return inputs, loss_mask


def _move_batch_to_device(inputs, loss_mask: Tensor, device):
    inputs = inputs.to(device)
    return inputs, loss_mask.to(device)


def _ce_loss_from_logits(logits: Tensor, input_ids: Tensor, loss_mask: Tensor):
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    label_ids = input_ids[:, 1:]
    shifted_mask = loss_mask[:, 1:].to(log_probs)
    token_log_probs = log_probs.gather(dim=-1, index=label_ids.unsqueeze(-1)).squeeze(-1)
    denom = shifted_mask.sum().clamp(min=1.0)
    return -(token_log_probs * shifted_mask).sum() / denom


def _kl_loss_from_logits(baseline_logits: Tensor, intervention_logits: Tensor, loss_mask: Tensor):
    baseline_probs = F.softmax(baseline_logits[:, :-1].to(torch.float64), dim=-1)
    intervention_log_probs = F.log_softmax(intervention_logits[:, :-1].to(torch.float64), dim=-1)
    shifted_mask = loss_mask[:, 1:].to(intervention_log_probs)
    kl_by_token = F.kl_div(intervention_log_probs, baseline_probs, reduction="none").sum(dim=-1)
    denom = shifted_mask.sum().clamp(min=1.0)
    return (kl_by_token * shifted_mask).sum() / denom


def _normalize_direction(direction: Tensor):
    return direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)


def _project_activation(activation: Tensor, direction: Tensor):
    direction = _normalize_direction(direction).to(activation)
    return activation - (activation @ direction).unsqueeze(-1) * direction


def _make_training_ablation_hooks(model_base: ModelBase, direction: Tensor):
    def input_hook(_module, inputs):
        activation = inputs[0] if isinstance(inputs, tuple) else inputs
        updated_activation = _project_activation(activation, direction)
        if isinstance(inputs, tuple):
            return (updated_activation, *inputs[1:])
        return updated_activation

    def output_hook(_module, _inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        updated_activation = _project_activation(activation, direction)
        if isinstance(output, tuple):
            return (updated_activation, *output[1:])
        return updated_activation

    fwd_pre_hooks = [
        (model_base.model_block_modules[layer], input_hook)
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (model_base.model_attn_modules[layer], output_hook)
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks.extend(
        (model_base.model_mlp_modules[layer], output_hook)
        for layer in range(model_base.model.config.num_hidden_layers)
    )
    return fwd_pre_hooks, fwd_hooks


def _make_training_addition_hooks(model_base: ModelBase, direction: Tensor, add_layer: int, alpha: Tensor):
    def input_hook(_module, inputs):
        activation = inputs[0] if isinstance(inputs, tuple) else inputs
        normalized_direction = _normalize_direction(direction).to(activation)
        updated_activation = activation + alpha.to(activation) * normalized_direction
        if isinstance(inputs, tuple):
            return (updated_activation, *inputs[1:])
        return updated_activation

    return [(model_base.model_block_modules[add_layer], input_hook)], []


def _forward_with_hooks(model_base: ModelBase, inputs, fwd_pre_hooks, fwd_hooks):
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        return model_base.model(**inputs).logits


def compute_direction_losses(
    model_base: ModelBase,
    records: List[Dict[str, str]],
    direction: Tensor,
    add_layer: int,
    alpha: Tensor,
):
    ablation_inputs, ablation_mask = tokenize_targets(
        model_base,
        [record["harmful_instruction"] for record in records],
        [record["ablation_target"] for record in records],
    )
    addition_inputs, addition_mask = tokenize_targets(
        model_base,
        [record["harmless_instruction"] for record in records],
        [record["addition_target"] for record in records],
    )
    retain_inputs, retain_mask = tokenize_targets(
        model_base,
        [record["harmless_instruction"] for record in records],
        [record["retain_target"] for record in records],
    )

    device = model_base.model.device
    ablation_inputs, ablation_mask = _move_batch_to_device(ablation_inputs, ablation_mask, device)
    addition_inputs, addition_mask = _move_batch_to_device(addition_inputs, addition_mask, device)
    retain_inputs, retain_mask = _move_batch_to_device(retain_inputs, retain_mask, device)

    ablation_pre_hooks, ablation_hooks = _make_training_ablation_hooks(model_base, direction)
    addition_pre_hooks, addition_hooks = _make_training_addition_hooks(model_base, direction, add_layer, alpha)

    ablation_logits = _forward_with_hooks(model_base, ablation_inputs, ablation_pre_hooks, ablation_hooks)
    addition_logits = _forward_with_hooks(model_base, addition_inputs, addition_pre_hooks, addition_hooks)
    with torch.no_grad():
        baseline_retain_logits = model_base.model(**retain_inputs).logits.detach()
    retain_logits = _forward_with_hooks(model_base, retain_inputs, ablation_pre_hooks, ablation_hooks)

    return {
        "ablation": _ce_loss_from_logits(ablation_logits, ablation_inputs["input_ids"], ablation_mask),
        "addition": _ce_loss_from_logits(addition_logits, addition_inputs["input_ids"], addition_mask),
        "retain": _kl_loss_from_logits(baseline_retain_logits, retain_logits, retain_mask),
    }


def sample_cone_coefficients(n_samples: int, cone_dim: int, device=None):
    coefficients = torch.rand(n_samples, cone_dim, device=device)
    return coefficients / coefficients.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def orthonormalize_basis(basis: Tensor):
    rows = []
    for row in basis:
        vector = row.clone()
        for previous in rows:
            vector = vector - torch.dot(vector, previous) * previous
        vector = vector / vector.norm().clamp(min=1e-8)
        rows.append(vector)
    return torch.stack(rows, dim=0)


def _initialize_basis(base_direction: Tensor, cone_dim: int, init_from_dim: bool):
    base = _normalize_direction(base_direction.detach().to(torch.float32).cpu()).squeeze()
    if cone_dim == 1:
        if init_from_dim:
            return base.unsqueeze(0)
        return _normalize_direction(torch.randn_like(base)).unsqueeze(0)

    rows = []
    if init_from_dim:
        rows.append(base)
    while len(rows) < cone_dim:
        rows.append(torch.randn_like(base))
    return orthonormalize_basis(torch.stack(rows, dim=0))


def _direction_from_basis(basis: Tensor, coefficients: Tensor):
    direction = coefficients.to(basis) @ basis
    return _normalize_direction(direction).squeeze(0)


def _project_direction_gradients(basis_parameter: torch.nn.Parameter):
    if basis_parameter.grad is None:
        return
    with torch.no_grad():
        normalized_basis = orthonormalize_basis(basis_parameter.data.detach())
        for idx in range(basis_parameter.shape[0]):
            grad = basis_parameter.grad[idx]
            for basis_vector in normalized_basis:
                grad = grad - torch.dot(grad, basis_vector) * basis_vector
            basis_parameter.grad[idx].copy_(grad)


def _iter_batches(records: List[Dict[str, str]], batch_size: int):
    for start_idx in range(0, len(records), batch_size):
        yield records[start_idx:start_idx + batch_size]


def optimize_refusal_geometry(
    cfg,
    model_base: ModelBase,
    harmful_train: List[str],
    harmless_train: List[str],
    base_direction: Float[Tensor, "d_model"],
    add_layer: int,
    artifact_dir: str,
):
    if cfg.direction_method not in ("rdo", "cone"):
        raise ValueError(f"Unsupported geometry optimization method: {cfg.direction_method}")

    cone_dim = cfg.rdo_cone_dim if cfg.direction_method == "cone" else 1
    optimization_cfg = GeometryOptimizationConfig(
        method=cfg.direction_method,
        cone_dim=cone_dim,
        epochs=cfg.rdo_epochs,
        batch_size=cfg.rdo_batch_size,
        effective_batch_size=cfg.rdo_effective_batch_size,
        learning_rate=cfg.rdo_learning_rate,
        target_max_new_tokens=cfg.rdo_target_max_new_tokens,
        n_cone_samples=cfg.rdo_n_cone_samples,
        ablation_lambda=cfg.rdo_ablation_lambda,
        addition_lambda=cfg.rdo_addition_lambda,
        retain_lambda=cfg.rdo_retain_lambda,
        init_from_dim=cfg.rdo_init_from_dim,
    )

    os.makedirs(artifact_dir, exist_ok=True)
    basis_path = os.path.join(artifact_dir, "basis.pt")
    direction_path = os.path.join(artifact_dir, "direction.pt")
    train_log_path = os.path.join(artifact_dir, "train_log.json")
    config_path = os.path.join(artifact_dir, "config.json")
    manifest_path = os.path.join(artifact_dir, "optimization_manifest.json")
    optimization_manifest = {
        "model_path": cfg.model_path,
        "harmful_train_signature": _instruction_list_signature(harmful_train),
        "harmless_train_signature": _instruction_list_signature(harmless_train),
        "base_direction_digest": _tensor_digest(base_direction),
        "add_layer": add_layer,
        "config": asdict(optimization_cfg),
    }

    if (
        cfg.reuse_artifacts
        and os.path.exists(direction_path)
        and os.path.exists(basis_path)
        and _manifest_matches(manifest_path, optimization_manifest)
    ):
        direction = torch.load(direction_path, map_location=model_base.model.device)
        basis = torch.load(basis_path, map_location="cpu")
        train_log = []
        if os.path.exists(train_log_path):
            with open(train_log_path, "r") as f:
                train_log = json.load(f)
        best_loss = train_log[-1]["total_loss"] if train_log else float("nan")
        wandb_log(
            {
                "geometry_refusal/best_loss": best_loss,
                "geometry_refusal/reused": 1,
                "geometry_refusal/cone_dim": cone_dim,
            }
        )
        return {
            "direction": direction,
            "basis": basis,
            "best_loss": best_loss,
            "train_log": train_log,
            "artifact_dir": artifact_dir,
            "config": asdict(optimization_cfg),
            "reused": True,
        }

    targets = generate_geometry_targets(
        cfg=cfg,
        model_base=model_base,
        harmful_instructions=harmful_train,
        harmless_instructions=harmless_train,
        base_direction=base_direction,
        add_layer=add_layer,
        artifact_dir=artifact_dir,
    )
    records = _build_training_records(targets)

    alpha = base_direction.detach().norm().to(model_base.model.device)
    initial_basis = _initialize_basis(base_direction, cone_dim, cfg.rdo_init_from_dim).to(model_base.model.device)
    basis_parameter = torch.nn.Parameter(initial_basis)
    optimizer = torch.optim.AdamW([basis_parameter], lr=cfg.rdo_learning_rate, betas=(0.9, 0.98), weight_decay=0.0)

    accumulation_steps = max(1, cfg.rdo_effective_batch_size // cfg.rdo_batch_size)
    train_log = []
    best_loss = float("inf")
    best_basis = initial_basis.detach().cpu()
    step_idx = 0

    for epoch in range(cfg.rdo_epochs):
        for batch_records in tqdm(_iter_batches(records, cfg.rdo_batch_size), desc=f"RDO epoch {epoch + 1}"):
            batch_losses = []
            sample_coefficients = []
            if cone_dim > 1 and cfg.rdo_n_cone_samples > 0:
                sample_coefficients = sample_cone_coefficients(
                    cfg.rdo_n_cone_samples,
                    cone_dim,
                    device=basis_parameter.device,
                )

            basis_coefficients = torch.eye(cone_dim, device=basis_parameter.device)
            all_coefficients = list(basis_coefficients) + list(sample_coefficients)
            for coefficients in all_coefficients:
                direction = _direction_from_basis(basis_parameter, coefficients.unsqueeze(0))
                losses = compute_direction_losses(model_base, batch_records, direction, add_layer, alpha)
                loss = (
                    cfg.rdo_ablation_lambda * losses["ablation"]
                    + cfg.rdo_addition_lambda * losses["addition"]
                    + cfg.rdo_retain_lambda * losses["retain"]
                )
                normalizer = max(1, len(all_coefficients))
                (loss / (normalizer * accumulation_steps)).backward()
                batch_losses.append({name: value.detach().item() for name, value in losses.items()})

            step_idx += 1
            if step_idx % accumulation_steps != 0:
                continue

            _project_direction_gradients(basis_parameter)
            torch.nn.utils.clip_grad_norm_([basis_parameter], 10.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                basis_parameter.data.copy_(orthonormalize_basis(basis_parameter.data))

            mean_losses = {
                key: sum(item[key] for item in batch_losses) / max(1, len(batch_losses))
                for key in ("ablation", "addition", "retain")
            }
            total_loss = (
                cfg.rdo_ablation_lambda * mean_losses["ablation"]
                + cfg.rdo_addition_lambda * mean_losses["addition"]
                + cfg.rdo_retain_lambda * mean_losses["retain"]
            )
            train_log.append(
                {
                    "step": step_idx,
                    "epoch": epoch + 1,
                    "total_loss": total_loss,
                    **mean_losses,
                }
            )
            wandb_log(
                {
                    "geometry_refusal/total_loss": total_loss,
                    "geometry_refusal/ablation_loss": mean_losses["ablation"],
                    "geometry_refusal/addition_loss": mean_losses["addition"],
                    "geometry_refusal/retain_loss": mean_losses["retain"],
                    "geometry_refusal/epoch": epoch + 1,
                },
                step=step_idx,
            )
            if total_loss < best_loss:
                best_loss = total_loss
                best_basis = basis_parameter.detach().cpu().clone()

    if not train_log:
        with torch.no_grad():
            basis_parameter.data.copy_(orthonormalize_basis(basis_parameter.data))
        best_basis = basis_parameter.detach().cpu().clone()
        best_loss = float("nan")

    torch.save(best_basis, basis_path)
    direction = best_basis[0].clone()
    torch.save(direction, direction_path)
    _write_json(train_log_path, train_log)
    _write_json(config_path, asdict(optimization_cfg))
    _write_manifest(manifest_path, optimization_manifest)
    wandb_log(
        {
            "geometry_refusal/best_loss": best_loss,
            "geometry_refusal/reused": 0,
            "geometry_refusal/cone_dim": cone_dim,
        }
    )
    wandb_artifact(
        name=f"{cfg.model_alias}-{cfg.direction_method}-geometry",
        artifact_type="refusal-direction",
        paths=[basis_path, direction_path, train_log_path, config_path, manifest_path],
    )

    return {
        "direction": direction.to(model_base.model.device),
        "basis": best_basis,
        "best_loss": best_loss,
        "train_log": train_log,
        "artifact_dir": artifact_dir,
        "config": asdict(optimization_cfg),
        "reused": False,
    }
