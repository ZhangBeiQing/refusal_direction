"""
=============================
Refusal Direction 主流程入口
=============================

本文件实现了论文 *Refusal in Language Models Is Mediated by a Single Direction* 的完整实验流程。

整体流水线分为以下阶段：
  1. 解析命令行参数 → 构建 Config 配置对象
  2. 加载并采样 harmful / harmless 数据集的 train/val/test 切分
  3. 对数据做 refusal 过滤（默认用 refusal score 过滤；可选 Nemotron judge 校准）
  4. 在 train 集上生成候选 refusal direction（mean_diffs）
  5. 在 val 集上选择最佳 direction（DIM 方法）
  6. （可选）用 RDO 或 Cone 方法进一步优化 direction
  7. 在评估数据集上生成 baseline / ablation / activation addition 三种 completion
  8. 对 completion 做越狱评估（jailbreak）和拒绝评估（refusal）
  9. 评估 CE loss 和困惑度变化

所有中间产物都有 manifest 缓存机制，可通过 --disable_artifact_cache 强制重跑。
"""

import argparse
import hashlib
import json
import os
import random

from dataset.load_dataset import load_dataset, load_dataset_split

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)
from pipeline.utils.logging import get_logger

from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.geometry_refusal import optimize_refusal_geometry
from pipeline.submodules.refusal_calibration import (
    cache_refusal_calibration_responses,
    derive_filtered_splits_and_refusal_toks,
    get_refusal_calibration_paths,
    load_judged_refusal_cache,
    run_refusal_judge_subprocess,
)
from pipeline.submodules.select_direction import get_refusal_scores, select_direction
from pipeline.utils.wandb_utils import wandb_log, wandb_run_context

logger = get_logger("RunPipeline")


# ============================================================================
# 工具函数：缓存校验与 manifest 管理
# ============================================================================
# 这些函数用于实现"增量构建"缓存机制。
# 每个中间产物（completions、evaluations、loss_evals 等）都伴有一个 manifest.json，
# 其中记录了生成该产物时所用的参数和输入数据的签名（SHA256 哈希）。
# 下次运行时，如果所有输入签名与 manifest 匹配，则直接复用缓存，跳过计算。


def _stable_digest(value):
    """
    对任意 JSON 可序列化的对象计算稳定 SHA256 摘要。
    
    原理：先转为 JSON 字符串（排序 key，关闭 ASCII 转义），再计算 SHA256。
    这样可以保证即使 dict 的 key 顺序不同，只要内容相同，摘要就一致。
    
    用途：为指令列表、数据集等计算"内容签名"，作为缓存判断依据。
    """
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _file_digest(path):
    """
    对大文件逐块计算 SHA256 摘要。
    
    逐块（1MB）读取是为了避免一次性将大文件（如 .pt / completion JSON）全部加载到内存。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path):
    """
    加载 manifest JSON 文件，返回内容字典。文件不存在则返回 None。
    """
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r") as f:
        return json.load(f)


def _manifest_matches(manifest_path, expected_payload):
    """
    检查磁盘上的 manifest 是否与期望的 payload 完全一致。
    一致 → 可复用缓存；不一致 → 需要重新生成。
    """
    return _load_manifest(manifest_path) == expected_payload


def _write_manifest(manifest_path, payload):
    """
    将 manifest payload 写入磁盘 JSON 文件。
    prettify（indent=4）方便人工排查问题。
    """
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _instruction_list_signature(instructions):
    """
    计算指令列表的内容签名。
    
    参数 instructions: list[str] — 每项是一条自然语言指令。
    返回: 列表排序后的 SHA256 摘要。
    """
    return _stable_digest(instructions)


def _dataset_signature(dataset):
    """
    计算数据集（list[dict]）的内容签名。
    
    每条样本只取 "instruction" 和 "category" 两个字段做规范化，
    避免无关字段（如生成时间戳）干扰缓存判断。
    """
    normalized = [
        {
            "instruction": row["instruction"],
            "category": row.get("category"),
        }
        for row in dataset
    ]
    return _stable_digest(normalized)


def _get_direction_signature(cfg, intervention_label):
    """
    获取当前 direction（方向向量）的签名信息。
    
    对于 "baseline" 干预（不加任何方向），签名只是一个标签。
    对于 "ablation" / "actadd"，签名包含：
      - direction.pt 文件的 SHA256 摘要
      - direction_metadata.json 的完整内容
    这样当方向改变时，缓存会自动失效。
    """
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


# ============================================================================
# 命令行参数解析
# ============================================================================


def parse_arguments():
    """
    解析所有命令行参数。
    
    所有参数都有默认值（定义在 Config 中），命令行传入的值会覆盖默认值。
    注意：--refusal_judge_model_path 为 None 时，跳过 refusal calibration 步骤。
    """
    parser = argparse.ArgumentParser(description="Run the refusal direction pipeline.")

    # ---- 模型路径 ----
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="目标模型在 HuggingFace 上的路径，例如 meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--refusal_judge_model_path", type=str, default=None,
        help="Nemotron refusal judge 模型的路径。不提供则使用默认 refusal score 过滤",
    )

    # ---- Refusal Judge 相关 ----
    parser.add_argument(
        "--refusal_judge_backend", type=str, choices=["vllm", "transformers"], default=None,
    )
    parser.add_argument(
        "--refusal_judge_gpu_memory_utilization", type=float, default=None,
    )

    # ---- 数据采样数量 ----
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)

    # ---- 生成参数 ----
    parser.add_argument("--max_new_tokens", type=int, default=None)

    # ---- CE Loss 评估参数 ----
    parser.add_argument("--ce_loss_n_batches", type=int, default=None)
    parser.add_argument("--ce_loss_batch_size", type=int, default=None)

    # ---- 批量大小 ----
    parser.add_argument("--activation_batch_size", type=int, default=None)
    parser.add_argument("--completion_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_batch_size", type=int, default=None)
    parser.add_argument("--refusal_calibration_max_new_tokens", type=int, default=None)

    # ---- 缓存控制 ----
    parser.add_argument(
        "--disable_refusal_calibration_cache", action="store_true",
        help="强制重新生成 refusal calibration 的响应和判定缓存",
    )
    parser.add_argument(
        "--disable_artifact_cache", action="store_true",
        help="强制重新生成所有中间产物（不复用缓存）",
    )

    # ---- Direction 方法选择 ----
    parser.add_argument(
        "--direction_method", choices=["dim", "rdo", "cone"], default=None,
        help="方向选择方法：dim=默认均值差方法, rdo=Refusal Direction Optimization, cone=Cone优化",
    )

    # ---- RDO / Cone 优化超参数 ----
    parser.add_argument("--rdo_cone_dim", type=int, default=None)
    parser.add_argument("--rdo_epochs", type=int, default=None)
    parser.add_argument("--rdo_batch_size", type=int, default=None)
    parser.add_argument("--rdo_effective_batch_size", type=int, default=None)
    parser.add_argument("--rdo_learning_rate", type=float, default=None)
    parser.add_argument("--rdo_target_max_new_tokens", type=int, default=None)
    parser.add_argument("--rdo_n_cone_samples", type=int, default=None)
    parser.add_argument("--rdo_ablation_lambda", type=float, default=None)
    parser.add_argument("--rdo_addition_lambda", type=float, default=None)
    parser.add_argument("--rdo_retain_lambda", type=float, default=None)
    parser.add_argument(
        "--rdo_random_init", action="store_true",
        help="RDO 方向不从 DIM 方向初始化，而是随机初始化",
    )

    # ---- Weights & Biases 集成 ----
    parser.add_argument("--wandb", action="store_true", help="启用 W&B 日志记录")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="逗号分隔的 W&B 标签")
    parser.add_argument("--wandb_dir", type=str, default=None)

    return parser.parse_args()


# ============================================================================
# Config 构建
# ============================================================================


def build_config_from_args(args):
    """
    从命令行参数和默认值构建 Config 对象。
    
    流程：
      1. 根据 model_path 的最后一截生成 model_alias（如 "Meta-Llama-3-8B-Instruct"）
      2. 用 model_alias 创建 Config 实例（自带所有默认值）
      3. 将命令行中显式传入的参数覆盖到 Config 上
      4. 处理 W&B 和 TOGETHER_API_KEY 相关的特殊逻辑
    """
    # 提取模型名称作为产物目录的别名
    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)

    # 这些字段可以直接从 args 覆盖到 cfg（名字一一对应）
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
        "direction_method",
        "rdo_cone_dim",
        "rdo_epochs",
        "rdo_batch_size",
        "rdo_effective_batch_size",
        "rdo_learning_rate",
        "rdo_target_max_new_tokens",
        "rdo_n_cone_samples",
        "rdo_ablation_lambda",
        "rdo_addition_lambda",
        "rdo_retain_lambda",
    ]
    for field_name in override_fields:
        value = getattr(args, field_name)
        if value is not None:
            setattr(cfg, field_name, value)

    # 下面这些字段的映射关系不完全是 1:1，需要特殊处理
    cfg.refusal_judge_model_path = args.refusal_judge_model_path
    if args.refusal_judge_backend is not None:
        cfg.refusal_judge_backend = args.refusal_judge_backend
    if args.disable_refusal_calibration_cache:
        cfg.reuse_refusal_calibration_cache = False
    if args.disable_artifact_cache:
        cfg.reuse_artifacts = False
    if args.rdo_random_init:
        cfg.rdo_init_from_dim = False

    # W&B 配置
    if args.wandb:
        cfg.wandb_enabled = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.wandb_entity = args.wandb_entity
    if args.wandb_mode is not None:
        cfg.wandb_mode = args.wandb_mode
    if args.wandb_name is not None:
        cfg.wandb_name = args.wandb_name
    if args.wandb_group is not None:
        cfg.wandb_group = args.wandb_group
    if args.wandb_tags is not None:
        cfg.wandb_tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
    if args.wandb_dir is not None:
        cfg.wandb_dir = args.wandb_dir

    # 如果没有设置 TOGETHER_API_KEY 环境变量，
    # 自动从 jailbreak 评估方法中移除 llamaguard2（因为它依赖 Together API）
    if os.environ.get("TOGETHER_API_KEY") is None and "llamaguard2" in cfg.jailbreak_eval_methodologies:
        cfg.jailbreak_eval_methodologies = tuple(
            m for m in cfg.jailbreak_eval_methodologies if m != "llamaguard2"
        )

    return cfg


# ============================================================================
# 步骤 1：加载并采样数据集
# ============================================================================


def load_and_sample_datasets(cfg):
    """
    从 dataset/splits/ 目录加载 harmful / harmless 的 train 和 val 切分，
    然后按 Config 中配置的数量随机采样。

    n_train / n_val 含义：
      -1 或更小值 → 使用该 split 的全部数据
      正数        → 随机采样该数量（固定种子 42）

    返回：
      harmful_train, harmless_train, harmful_val, harmless_val
      每个都是 list[str]（指令文本列表）

    注意：固定随机种子 42，保证每次采样结果一致。
    """
    random.seed(42)

    # harmful 和 harmless 是论文中定义的两类指令：
    #   harmful_train: 用于提取 refusal direction 的"有害"指令
    #   harmless_train: 用于对比的"无害"指令
    def _sample(dataset, n):
        return random.sample(dataset, n) if n > 0 else dataset

    harmful_train = _sample(
        load_dataset_split(harmtype="harmful", split="train", instructions_only=True),
        cfg.n_train,
    )
    harmless_train = _sample(
        load_dataset_split(harmtype="harmless", split="train", instructions_only=True),
        cfg.n_train,
    )
    harmful_val = _sample(
        load_dataset_split(harmtype="harmful", split="val", instructions_only=True),
        cfg.n_val,
    )
    harmless_val = _sample(
        load_dataset_split(harmtype="harmless", split="val", instructions_only=True),
        cfg.n_val,
    )
    return harmful_train, harmless_train, harmful_val, harmless_val


# ============================================================================
# 步骤 2：数据过滤（refusal score 或 Nemotron judge 校准）
# ============================================================================


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    使用 refusal score（基于模型内部 refusal token 的 logits）过滤数据。
    
    - 对于 harmful 数据：只保留 refusal score > 0 的样本
      （模型确实在拒绝该指令，说明该指令是"有效有害"的）
    - 对于 harmless 数据：只保留 refusal score < 0 的样本
      （模型没有拒绝该指令，说明该指令确实是"无害"的）

    这一步是为了提高方向的纯度：如果某个 harmful 指令模型根本没拒绝，
    那它的中间表示对提取 refusal direction 没有帮助。
    """
    def filter_examples(dataset, scores, threshold, comparison):
        """通用过滤器：按 score 和阈值筛选"""
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        # 对 train 集计算 refusal score
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
        # harmful 保留 score > 0 的（越正说明拒绝越强）
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        # harmless 保留 score < 0 的（越负说明越不拒绝）
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        # 对 val 集同样计算 refusal score 并过滤
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
    """
    使用外部 Nemotron judge 模型进行更精确的 refusal 校准。
    
    这是 data filtering 的高级替代方案，分为三步：
    
    1. 用目标模型生成所有 train/val 指令的响应（response cache）
       - 有 manifest 缓存，默认复用
    2. 用 Nemotron judge 模型对每个响应做"是否拒绝"的判断（judged cache）
       - 这一步需要 GPU 资源独立运行（原模型会被临时卸载）
       - 有 manifest 缓存，默认复用
    3. 根据 judged 结果：
       - 过滤掉 judge 判定不一致的样本
       - 重新确定最优的 refusal token IDs（通过词频统计）

    返回：
      过滤后的四个 split + 更新了 refusal_toks 的 model_base
    """
    # 整理所有 split 的指令，方便统一处理
    split_to_instructions = {
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
    }
    calibration_paths = get_refusal_calibration_paths(cfg)
    os.makedirs(calibration_paths["artifact_dir"], exist_ok=True)

    # ---- 第 1 步：生成/复用 response cache ----
    response_cache_exists = os.path.exists(calibration_paths["response_cache_path"])
    judged_cache_exists = os.path.exists(calibration_paths["judged_cache_path"])

    # response cache 的 manifest：模型路径 + 所有 split 的指令签名 + max_new_tokens
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
        logger.info("  [Step 3a] 生成响应缓存 (completion_batch_size=%d, max_new_tokens=%d)...",
                    cfg.completion_batch_size, cfg.refusal_calibration_max_new_tokens)
        # 用目标模型逐批生成所有指令的响应，写入 JSON 文件
        cache_refusal_calibration_responses(
            model_base=model_base,
            split_to_instructions=split_to_instructions,
            output_path=calibration_paths["response_cache_path"],
            batch_size=cfg.completion_batch_size,
            max_new_tokens=cfg.refusal_calibration_max_new_tokens,
        )
        _write_manifest(calibration_paths["response_cache_manifest_path"], response_cache_manifest)
        response_cache_exists = True
    else:
        logger.info("  [Step 3a] 复用已有响应缓存: %s", calibration_paths["response_cache_path"])

    # ---- 第 2 步：生成/复用 judged cache ----
    judged_cache_manifest = {
        "judge_model_path": cfg.refusal_judge_model_path,
        "judge_backend": cfg.refusal_judge_backend,
        "response_cache_file_digest": (
            _file_digest(calibration_paths["response_cache_path"])
            if response_cache_exists
            else None
        ),
    }

    need_judged_cache = not (
        cfg.reuse_refusal_calibration_cache
        and judged_cache_exists
        and _manifest_matches(calibration_paths["judged_cache_manifest_path"], judged_cache_manifest)
    )

    if need_judged_cache:
        logger.info("  [Step 3b] 运行 Nemotron judge (backend=%s, batch_size=%d)...",
                    cfg.refusal_judge_backend, cfg.refusal_calibration_batch_size)
        # judge 模型和原模型同时占用显存可能 OOM，先卸载原模型
        model_base.del_model()
        # 在子进程中运行 judge 模型，完成后释放其显存
        run_refusal_judge_subprocess(
            cfg=cfg,
            input_path=calibration_paths["response_cache_path"],
            output_path=calibration_paths["judged_cache_path"],
        )
        _write_manifest(calibration_paths["judged_cache_manifest_path"], judged_cache_manifest)
        # judge 完成后重新加载原模型
        logger.info("  [Step 3b] Judge 完成，重新加载目标模型...")
        model_base = construct_model_base(cfg.model_path)
    else:
        logger.info("  [Step 3b] 复用已有 judge 缓存: %s", calibration_paths["judged_cache_path"])

    # ---- 第 3 步：根据 judged 结果过滤样本并确定 refusal tokens ----
    logger.info("  [Step 3c] 过滤样本并确定 refusal tokens...")
    judged_payload = load_judged_refusal_cache(calibration_paths["judged_cache_path"])
    split_to_filtered_instructions, refusal_toks, summary = (
        derive_filtered_splits_and_refusal_toks(
            judged_payload=judged_payload,
            tokenizer=model_base.tokenizer,
            cfg=cfg,
            fallback_refusal_toks=model_base.refusal_toks,
        )
    )
    logger.info("  [Step 3c] refusal_toks=%s", refusal_toks)

    # 保存校准摘要（包含每个 split 的过滤统计、选中的 refusal tokens 等信息）
    with open(calibration_paths["summary_path"], "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 更新 model_base 的 refusal_toks
    model_base.refusal_toks = refusal_toks

    # 返回过滤后的数据，如果某个 split 没有过滤结果就用原始数据
    return (
        split_to_filtered_instructions.get("harmful_train", harmful_train),
        split_to_filtered_instructions.get("harmless_train", harmless_train),
        split_to_filtered_instructions.get("harmful_val", harmful_val),
        split_to_filtered_instructions.get("harmless_val", harmless_val),
        model_base,
    )


# ============================================================================
# 步骤 3：生成候选 refusal direction（mean_diffs）
# ============================================================================


def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """
    对模型每一层、每一个 token 位置，计算 harmful 和 harmless 指令的
    中间表示（残差流激活）的均值差向量。
    
    这就是论文中最基本的 candidate direction：
      direction = mean(activations_harmful) - mean(activations_harmless)
    
    产物保存在 artifact_path/generate_directions/mean_diffs.pt，
    格式为 dict[layer][pos] = tensor(d_model,)

    有 manifest 缓存机制，如果输入指令列表不变则复用。
    """
    import torch

    artifact_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(artifact_dir, exist_ok=True)
    mean_diffs_path = os.path.join(artifact_dir, "mean_diffs.pt")
    manifest_path = os.path.join(artifact_dir, "manifest.json")

    # manifest 记录模型路径和训练指令签名
    manifest = {
        "model_path": cfg.model_path,
        "harmful_train_signature": _instruction_list_signature(harmful_train),
        "harmless_train_signature": _instruction_list_signature(harmless_train),
    }

    # 检查缓存
    if cfg.reuse_artifacts and os.path.exists(mean_diffs_path) and _manifest_matches(manifest_path, manifest):
        print(f"Reusing cached candidate directions from {mean_diffs_path}")
        wandb_log({"artifact/generate_directions_reused": 1})
        return torch.load(mean_diffs_path, map_location=model_base.model.device)

    # 实际计算：逐层、逐位置收集 harmful/harmless 激活，计算均值差
    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=artifact_dir,
        batch_size=cfg.activation_batch_size,
    )

    _write_manifest(manifest_path, manifest)
    wandb_log({"artifact/generate_directions_reused": 0})
    return mean_diffs


# ============================================================================
# 步骤 4：选择最佳 direction（DIM 方法）
# ============================================================================


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """
    在验证集上评估所有候选方向（不同层、不同位置），选择最优的 refusal direction。
    
    评估方式（DIM - Difference in Means）：
      对于每个候选方向，在 harmful_val 和 harmless_val 上分别测量消融效果，
      选择"消除 harmful 拒绝效果最好、同时对 harmless 影响最小"的方向。
    
    返回值：
      pos: 该方向对应的 token 位置索引
      layer: 该方向对应的模型层索引
      direction: 选择出的方向向量 tensor(d_model,)
    
    产物包括：
      - dim_direction.pt / dim_direction_metadata.json：DIM 选出的方向
      - 如果最终方向方法就是 dim，同时保存为 artifact_path/direction.pt
    """
    import torch

    artifact_dir = os.path.join(cfg.artifact_path(), "select_direction")
    os.makedirs(artifact_dir, exist_ok=True)
    direction_path = os.path.join(cfg.artifact_path(), "direction.pt")
    direction_metadata_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    dim_direction_path = os.path.join(artifact_dir, "dim_direction.pt")
    dim_direction_metadata_path = os.path.join(artifact_dir, "dim_direction_metadata.json")
    manifest_path = os.path.join(artifact_dir, "manifest.json")
    mean_diffs_path = os.path.join(cfg.artifact_path(), "generate_directions", "mean_diffs.pt")

    manifest = {
        "model_path": cfg.model_path,
        "harmful_val_signature": _instruction_list_signature(harmful_val),
        "harmless_val_signature": _instruction_list_signature(harmless_val),
        "candidate_directions_file_digest": (
            _file_digest(mean_diffs_path) if os.path.exists(mean_diffs_path) else None
        ),
    }

    # 尝试复用缓存：先查专属的 dim_direction 文件
    if (
        cfg.reuse_artifacts
        and os.path.exists(dim_direction_path)
        and os.path.exists(dim_direction_metadata_path)
        and _manifest_matches(manifest_path, manifest)
    ):
        print(f"Reusing cached DIM selected direction from {dim_direction_path}")
        with open(dim_direction_metadata_path, "r") as f:
            metadata = json.load(f)
        direction = torch.load(dim_direction_path, map_location=model_base.model.device)
        wandb_log({
            "artifact/select_direction_reused": 1,
            "direction/layer": metadata["layer"],
            "direction/pos": metadata["pos"],
        })
        return metadata["pos"], metadata["layer"], direction

    # 再查通用的 direction.pt（可能已被 RDO/Cone 覆盖，但 DIM 阶段的数据还在 metadata 里）
    if (
        cfg.reuse_artifacts
        and os.path.exists(direction_path)
        and os.path.exists(direction_metadata_path)
        and _manifest_matches(manifest_path, manifest)
    ):
        with open(direction_metadata_path, "r") as f:
            metadata = json.load(f)
        # 只有纯 DIM 方法才从 direction.pt 恢复
        if metadata.get("method", "dim") == "dim":
            print(f"Reusing cached selected direction from {direction_path}")
            direction = torch.load(direction_path, map_location=model_base.model.device)
            # 同时恢复 dim_direction 文件以便后续单独使用
            with open(dim_direction_metadata_path, "w") as f:
                json.dump({"pos": metadata["pos"], "layer": metadata["layer"], "method": "dim"}, f, indent=4)
            torch.save(direction, dim_direction_path)
            wandb_log({
                "artifact/select_direction_reused": 1,
                "direction/layer": metadata["layer"],
                "direction/pos": metadata["pos"],
            })
            return metadata["pos"], metadata["layer"], direction

    # 缓存未命中，实际运行 select_direction
    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=artifact_dir,
        batch_size=cfg.activation_batch_size,
    )

    # 保存 DIM 阶段的产物
    dim_metadata = {"pos": pos, "layer": layer, "method": "dim"}
    with open(dim_direction_metadata_path, "w") as f:
        json.dump(dim_metadata, f, indent=4)
    torch.save(direction, dim_direction_path)

    # 如果最终方法就是 DIM，直接保存为最终 direction
    if cfg.direction_method == "dim":
        with open(direction_metadata_path, "w") as f:
            json.dump(dim_metadata, f, indent=4)
        torch.save(direction, direction_path)

    _write_manifest(manifest_path, manifest)
    wandb_log({
        "artifact/select_direction_reused": 0,
        "direction/layer": layer,
        "direction/pos": pos,
    })

    return pos, layer, direction


# ============================================================================
# 步骤 5（可选）：用 RDO / Cone 方法优化 direction
# ============================================================================


def optimize_and_save_geometry_direction(cfg, model_base, harmful_train, harmless_train, pos, layer, dim_direction):
    """
    在 DIM 方向基础上，用 RDO (Refusal Direction Optimization) 或 Cone 方法
    进一步优化 refusal direction。
    
    RDO 方法：
      通过梯度优化，找到一个方向使得：
      1. 消融该方向时 harmful 指令不再被拒绝（ablation loss ↓）
      2. 增强该方向时 harmless 指令也产生拒绝（addition loss ↓）  
      3. 正常无害指令不受影响（retain loss ↓）
    
    Cone 方法：
      在 DIM 方向附近采样多个候选方向的锥形区域，选择最优。

    结果保存为最终的 artifact_path/direction.pt。
    """
    import torch

    artifact_dir = os.path.join(cfg.artifact_path(), "geometry_refusal", cfg.direction_method)
    result = optimize_refusal_geometry(
        cfg=cfg,
        model_base=model_base,
        harmful_train=harmful_train,
        harmless_train=harmless_train,
        base_direction=dim_direction,
        add_layer=layer,
        artifact_dir=artifact_dir,
    )

    # 保存最终方向和元数据
    direction_path = os.path.join(cfg.artifact_path(), "direction.pt")
    direction_metadata_path = os.path.join(cfg.artifact_path(), "direction_metadata.json")
    metadata = {
        "method": cfg.direction_method,
        "pos": pos,
        "layer": layer,
        "source_method": "dim",  # 记录初始方向来源于 DIM
        "geometry_artifact_dir": artifact_dir,
        "best_loss": result["best_loss"],
        "reused": result["reused"],
        "config": result["config"],
    }
    if cfg.direction_method == "cone":
        metadata["cone_dim"] = int(result["basis"].shape[0])
        metadata["direction_basis_index"] = 0

    torch.save(result["direction"], direction_path)
    with open(direction_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    wandb_log({
        "direction/layer": layer,
        "direction/pos": pos,
        "geometry_refusal/best_loss": result["best_loss"],
        "geometry_refusal/reused": int(result["reused"]),
    })

    return layer, result["direction"]


# ============================================================================
# 步骤 6：生成三种配置的 completions
# ============================================================================


def generate_and_save_completions_for_dataset(
    cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None
):
    """
    对指定数据集生成模型 completion，支持三种干预模式：
      - "baseline": 不做任何干预，原样生成
      - "ablation": 从残差流中消融（减去）refusal direction
      - "actadd":  向残差流中添加 refusal direction（逆转方向系数 -1.0）

    生成的 completions 保存为 JSON 文件。
    有 manifest 缓存机制。
    """
    completions_dir = os.path.join(cfg.artifact_path(), "completions")
    os.makedirs(completions_dir, exist_ok=True)
    completions_path = os.path.join(
        completions_dir, f"{dataset_name}_{intervention_label}_completions.json"
    )
    manifest_path = os.path.join(
        completions_dir, f"{dataset_name}_{intervention_label}_completions_manifest.json"
    )

    # 如果未传入 dataset，则从文件加载
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

    # 实际生成：逐批传入 hook，产生 completion
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


# ============================================================================
# 步骤 7：对 completions 做越狱/拒绝评估
# ============================================================================


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """
    加载已生成的 completions，运行越狱评估（jailbreak evaluation）。
    
    评估方法（eval_methodologies）包括：
      - "substring_matching": 基于子串匹配（匹配 "I cannot"/"I apologize" 等拒绝短语）
      - "llamaguard2": 使用 Llama Guard 2 模型评估（需要 TOGETHER_API_KEY）

    结果保存为 *_evaluations.json。
    """
    evaluations_path = os.path.join(
        cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"
    )
    completions_path = os.path.join(
        cfg.artifact_path(), f"completions/{dataset_name}_{intervention_label}_completions.json"
    )
    manifest_path = os.path.join(
        cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations_manifest.json"
    )

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


# ============================================================================
# 步骤 8：评估 CE Loss（困惑度变化）
# ============================================================================


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """
    评估在 harmless 文本上应用方向干涉前后的 CE Loss 变化。
    
    主要目的是验证：消融/添加 refusal direction 后，
    模型在正常无害文本上的语言建模能力没有显著退化（困惑度不会飙升）。
    
    以 baseline 生成的 harmless completions 作为评估文本，
    分别计算 baseline / ablation / actadd 三种配置下的 CE loss。
    """
    from pipeline.submodules.evaluate_loss import evaluate_loss

    loss_eval_dir = os.path.join(cfg.artifact_path(), "loss_evals")
    os.makedirs(loss_eval_dir, exist_ok=True)
    loss_eval_path = os.path.join(loss_eval_dir, f"{intervention_label}_loss_eval.json")
    manifest_path = os.path.join(loss_eval_dir, f"{intervention_label}_loss_eval_manifest.json")

    on_distribution_completions_file_path = os.path.join(
        cfg.artifact_path(), "completions/harmless_baseline_completions.json"
    )

    manifest = {
        "model_path": cfg.model_path,
        "intervention_label": intervention_label,
        "direction_signature": _get_direction_signature(cfg, intervention_label),
        "ce_loss_batch_size": cfg.ce_loss_batch_size,
        "ce_loss_n_batches": cfg.ce_loss_n_batches,
        "on_distribution_completions_digest": (
            _file_digest(on_distribution_completions_file_path)
            if os.path.exists(on_distribution_completions_file_path)
            else None
        ),
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


# ============================================================================
# 主流程：run_pipeline()
# ============================================================================


def run_pipeline(cfg: Config):
    """
    串联完整实验流程的主函数。
    
    整体流程如下（==> 表示步骤间的数据流向）：

    ┌─────────────────────────────────────────────────────────────┐
    │ 1. 加载模型适配器 (construct_model_base)                      │
    │ 2. 加载并采样数据集 (load_and_sample_datasets)                │
    │ 3. 数据过滤 ──┬── 简单模式: filter_data (refusal score)       │
    │                └── 校准模式: calibrate_refusal_proxy (judge)  │
    │ 4. 生成候选方向 (generate_and_save_candidate_directions)      │
    │ 5. 选择最佳方向 (select_and_save_direction, DIM)              │
    │ 6. [可选] 优化方向 (optimize_and_save_geometry_direction)     │
    │ 7. 构造三种 hook 配置:                                        │
    │    - baseline:  空 hook (对照组)                              │
    │    - ablation:  消融 refusal direction (去除拒绝行为)         │
    │    - actadd:    反向添加 direction (诱导拒绝)                  │
    │ 8. 对每个评估数据集，用三种配置生成 completion                 │
    │ 9. 对每个评估数据集，评估 jailbreak 指标                       │
    │ 10. 对 harmless 数据，额外评估:                                │
    │     - baseline 的 denial 率                                   │
    │     - actadd (coeff=+1.0) 是否会让无害指令也被拒绝             │
    │ 11. 对三种配置评估 CE Loss                                     │
    └─────────────────────────────────────────────────────────────┘
    """
    # 启动 W&B run（如果启用），整个流水线在同一个 run 上下文中
    with wandb_run_context(cfg):
        # ---- 步骤 1：加载模型适配器 ----
        # construct_model_base 根据 model_path 自动识别模型族 (Llama/Qwen/GLM/Gemma/Yi)
        # 并返回适配后的 ModelBase 子类实例，包含 tokenizer、hook 注册、生成接口等
        model_base = construct_model_base(cfg.model_path)

        # ---- 步骤 2：加载并采样数据集 ----
        harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

        # ---- 步骤 3：数据过滤 ----
        # 二选一路径：
        #   A) 如果提供了 refusal_judge_model_path → 用 Nemotron judge 校准
        #   B) 否则 → 用 refusal score 简单过滤
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

        # ---- 步骤 4：生成候选 refusal direction ----
        # 返回 mean_diffs: dict[layer][pos] → tensor(d_model,)
        candidate_directions = generate_and_save_candidate_directions(
            cfg, model_base, harmful_train, harmless_train
        )

        # ---- 步骤 5：在 val 集上选择最佳方向 (DIM) ----
        # 返回最优的 pos、layer 索引和方向向量
        pos, layer, direction = select_and_save_direction(
            cfg, model_base, harmful_val, harmless_val, candidate_directions
        )

        # ---- 步骤 6（可选）：RDO / Cone 优化方向 ----
        if cfg.direction_method in ("rdo", "cone"):
            layer, direction = optimize_and_save_geometry_direction(
                cfg,
                model_base,
                harmful_train,
                harmless_train,
                pos,
                layer,
                direction,
            )

        # ---- 步骤 7：构造三种干预配置的 hook ----
        # baseline: 无干预，模型原样输出
        baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

        # ablation: 从指定层减去 refusal direction，消除拒绝行为
        # get_all_direction_ablation_hooks 为每一层注册一个减去方向的 hook
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(
            model_base, direction
        )

        # actadd: 在指定层加入"反向" direction (coeff=-1.0)
        # 论文中这会诱导模型对 harmful 指令产生拒绝行为
        # 注意：这里的 -1.0 是相对于 ablation 的减法方向而言的"反向"
        actadd_fwd_pre_hooks, actadd_fwd_hooks = [
            (
                model_base.model_block_modules[layer],
                get_activation_addition_input_pre_hook(
                    vector=direction, coeff=-1.0
                ),
            )
        ], []

        # ---- 步骤 8：对每个评估数据集生成三种 completion ----
        for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(
                cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks,
                "baseline", dataset_name,
            )
            generate_and_save_completions_for_dataset(
                cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks,
                "ablation", dataset_name,
            )
            generate_and_save_completions_for_dataset(
                cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks,
                "actadd", dataset_name,
            )

        # ---- 步骤 9：对每个评估数据集做 jailbreak 评估 ----
        for dataset_name in cfg.evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(
                cfg, "baseline", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies,
            )
            evaluate_completions_and_save_results_for_dataset(
                cfg, "ablation", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies,
            )
            evaluate_completions_and_save_results_for_dataset(
                cfg, "actadd", dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies,
            )

        # ---- 步骤 10：对 harmless 数据做额外评估 ----
        # 目的：验证在 harmless 指令上添加 refusal direction 是否会导致
        # 模型错误地拒绝本来无害的请求（即"过度拒绝"问题）
        harmless_test = random.sample(
            load_dataset_split(harmtype="harmless", split="test"), cfg.n_test
        )

        # baseline: 基准无害生成
        generate_and_save_completions_for_dataset(
            cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks,
            "baseline", "harmless", dataset=harmless_test,
        )

        # actadd refusal: coeff=+1.0 — 正向添加 direction，看是否会诱导拒绝
        actadd_refusal_pre_hooks, actadd_refusal_hooks = [
            (
                model_base.model_block_modules[layer],
                get_activation_addition_input_pre_hook(
                    vector=direction, coeff=+1.0
                ),
            )
        ], []
        generate_and_save_completions_for_dataset(
            cfg, model_base, actadd_refusal_pre_hooks, actadd_refusal_hooks,
            "actadd", "harmless", dataset=harmless_test,
        )

        # 评估 harmless 场景的拒绝率
        evaluate_completions_and_save_results_for_dataset(
            cfg, "baseline", "harmless", eval_methodologies=cfg.refusal_eval_methodologies,
        )
        evaluate_completions_and_save_results_for_dataset(
            cfg, "actadd", "harmless", eval_methodologies=cfg.refusal_eval_methodologies,
        )

        # ---- 步骤 11：评估 CE Loss ----
        # 分别对 baseline / ablation / actadd 三种配置计算交叉熵损失
        evaluate_loss_for_datasets(
            cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, "baseline",
        )
        evaluate_loss_for_datasets(
            cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, "ablation",
        )
        evaluate_loss_for_datasets(
            cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, "actadd",
        )


# ============================================================================
# 入口：解析参数 → 构建 Config → 运行主流程
# ============================================================================
if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(build_config_from_args(args))
