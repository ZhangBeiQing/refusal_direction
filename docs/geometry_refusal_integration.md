# Geometry of Refusal 集成说明

## 背景

本次集成参考论文 *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence* 以及本地克隆的 `geometry-of-refusal` 实现。

主项目原本实现的是 DIM refusal direction 流程：

1. harmful / harmless 激活均值差生成候选方向
2. 在验证集上选择最佳 direction
3. 对 baseline / ablation / actadd 生成 completion 并评估
4. 评估 CE loss / PPL

新论文中的核心新增算法是：

- RDO: Refusal Direction Optimization，通过梯度优化单个 refusal direction
- RCO / cone: Refusal Cone Optimization，通过正交 basis 表示多维 refusal cone

本次没有直接搬 `geometry-of-refusal` 的 `nnsight + wandb` 运行方式，而是改造成主项目已有的 HuggingFace model wrapper 与 forward hook 机制。

## 新增内容

### 1. 新增 `pipeline/submodules/geometry_refusal.py`

该模块实现了 HuggingFace 版本的 RDO / cone 训练逻辑：

- 使用已有 DIM direction 生成训练目标
  - harmful prompt + DIM ablation 生成 `ablation_target`
  - harmless prompt + DIM activation addition 生成 `addition_target`
  - harmless prompt baseline 生成 `retain_target`
- 用梯度优化 direction / cone basis
  - `ablation_loss`: 对 harmful prompt 做 direction ablation 后拟合 `ablation_target`
  - `addition_loss`: 对 harmless prompt 做 activation addition 后拟合 refusal target
  - `retain_loss`: 对 harmless prompt 做 ablation 后约束 KL，不破坏 baseline 输出
- cone 模式会维护正交 basis，并从正 cone 中采样方向参与优化

### 2. `run_pipeline.py` 新增可选算法入口

新增参数：

```bash
--direction_method dim|rdo|cone
--rdo_cone_dim <int>
--rdo_epochs <int>
--rdo_batch_size <int>
--rdo_effective_batch_size <int>
--rdo_learning_rate <float>
--rdo_target_max_new_tokens <int>
--rdo_n_cone_samples <int>
--rdo_ablation_lambda <float>
--rdo_addition_lambda <float>
--rdo_retain_lambda <float>
--rdo_random_init
```

默认仍是：

```bash
--direction_method dim
```

因此原来的复现实验行为不会被改变。

### 3. DIM 选择结果与最终 direction 解耦

原流程会直接把 DIM 选出的方向写到：

```text
pipeline/runs/<model_alias>/direction.pt
pipeline/runs/<model_alias>/direction_metadata.json
```

现在额外保留 DIM 中间结果：

```text
pipeline/runs/<model_alias>/select_direction/dim_direction.pt
pipeline/runs/<model_alias>/select_direction/dim_direction_metadata.json
```

当 `direction_method=rdo|cone` 时，DIM direction 作为 RDO/RCO 的初始化和 target generation 基准，优化后的方向再写回根目录 `direction.pt`，后续 completion/eval/loss 继续复用原链路。

### 4. 新增 RDO/RCO 产物目录

RDO 产物：

```text
pipeline/runs/<model_alias>/geometry_refusal/rdo/
```

cone 产物：

```text
pipeline/runs/<model_alias>/geometry_refusal/cone/
```

主要文件：

- `targets.json`: RDO/RCO 训练目标
- `targets_manifest.json`: target generation 缓存签名
- `basis.pt`: 单向量 RDO 时 shape 为 `[1, d_model]`，cone 时 shape 为 `[cone_dim, d_model]`
- `direction.pt`: 当前主流程使用的代表方向
- `train_log.json`: 每个优化 step 的 loss
- `optimization_manifest.json`: 训练缓存签名

cone 模式下，主项目当前的 completion/eval/loss 仍然是单向量接口，所以默认使用 `basis[0]` 作为代表方向写入根目录 `direction.pt`。完整 cone basis 会保存在 `basis.pt`。

## Qwen3.5 + Nemotron 运行方式

推荐主模型：

```text
/root/autodl-tmp/Qwen3.5-4B
```

推荐 refusal judge：

```text
/root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B
```

RDO smoke test：

```bash
source /root/venv/refuse/bin/activate
export HF_ENDPOINT="https://hf-mirror.com"

python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --direction_method rdo \
  --n_train 8 \
  --n_val 4 \
  --n_test 8 \
  --max_new_tokens 128 \
  --ce_loss_n_batches 2 \
  --ce_loss_batch_size 1 \
  --activation_batch_size 1 \
  --completion_batch_size 1 \
  --refusal_calibration_batch_size 2 \
  --refusal_calibration_max_new_tokens 64 \
  --rdo_epochs 1 \
  --rdo_batch_size 1 \
  --rdo_effective_batch_size 1 \
  --rdo_target_max_new_tokens 16
```

cone smoke test：

```bash
python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --direction_method cone \
  --rdo_cone_dim 2 \
  --rdo_n_cone_samples 1 \
  --n_train 8 \
  --n_val 4 \
  --n_test 8 \
  --max_new_tokens 128 \
  --ce_loss_n_batches 2 \
  --ce_loss_batch_size 1 \
  --activation_batch_size 1 \
  --completion_batch_size 1 \
  --refusal_calibration_batch_size 2 \
  --refusal_calibration_max_new_tokens 64 \
  --rdo_epochs 1 \
  --rdo_batch_size 1 \
  --rdo_effective_batch_size 1 \
  --rdo_target_max_new_tokens 16
```

## Nemotron 的使用位置

Nemotron 仍然只用于 refusal calibration：

1. Qwen3.5 先生成 train/val split 的 baseline responses
2. Nemotron 判断每条 response 是否是 refusal
3. 主项目根据 judge 结果过滤 harmful / harmless split
4. 主项目根据 first response token 重新校准 `refusal_toks`
5. RDO/RCO 和 `select_direction` 继续使用这个校准后的快速 refusal-token proxy

没有把 Nemotron 放进 RDO/RCO 内层训练循环，因为那会导致每个候选方向都要生成完整回答再调用 judge，成本过高。

## 显存说明

RTX 5090 32GB 理论上足够跑 4B 级 Qwen3.5 的 DIM/RDO 小规模 smoke test。当前主流程在 Nemotron judge 阶段会释放目标模型，再由子进程加载 judge 模型，避免两个 4B 模型同时常驻显存。

RDO/RCO 比 DIM 更吃显存和时间，建议先用：

- `--rdo_batch_size 1`
- `--rdo_effective_batch_size 1`
- `--rdo_target_max_new_tokens 16`
- `--rdo_epochs 1`

确认链路能跑通后再扩大规模。

## W&B on AutoDL / SSH

本项目已经添加可选 W&B 控制层。默认不启用，避免影响复现实验；启用时默认 `offline`，日志写到 `/root/autodl-tmp/wandb`，不会占用较小的系统盘。

服务器不需要打开浏览器。在线模式下，训练进程只负责把日志发到 W&B cloud；dashboard URL 可以在你本地电脑浏览器打开。

### 推荐：先离线记录

```bash
python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --direction_method rdo \
  --wandb \
  --wandb_project refusal-direction \
  --wandb_mode offline
```

离线 run 会保存在：

```text
/root/autodl-tmp/wandb/wandb/offline-run-*
```

之后如果服务器能访问 `wandb.ai`，在服务器上同步：

```bash
source /root/venv/refuse/bin/activate
export WANDB_API_KEY="<your_wandb_api_key>"
wandb sync /root/autodl-tmp/wandb/wandb/offline-run-*
```

也可以把 `/root/autodl-tmp/wandb/wandb/offline-run-*` 下载到本地电脑，在本地装好 wandb 后执行 `wandb sync`。

### 在线模式

如果 AutoDL 服务器能稳定访问 W&B：

```bash
export WANDB_API_KEY="<your_wandb_api_key>"

python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --direction_method rdo \
  --wandb \
  --wandb_project refusal-direction \
  --wandb_mode online
```

如果在线模式卡住或网络不稳定，切回 `--wandb_mode offline`。

### 当前记录内容

启用 W&B 后会记录：

- 完整 `Config`
- DIM direction 是否复用缓存
- 选中 direction 的 layer / position
- RDO / cone 的 total / ablation / addition / retain loss
- RDO / cone 最优 loss 与是否复用缓存
- RDO / cone 训练产物 artifact

## 测试

新增轻量单元测试：

```text
tests/test_geometry_refusal.py
```

覆盖内容：

- cone 系数采样为非负单位向量
- Gram-Schmidt basis 正交化
- CE loss 的 shifted target mask
- retain KL loss 对 identical logits 为 0
- prompt / target loss mask 的 EOI token 与 fallback 行为
