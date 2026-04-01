# Refusal Direction 复现与推理说明

本仓库用于复现论文 *Refusal in Language Models Is Mediated by a Single Direction*，并在当前代码基础上扩展了：

- `Qwen3.5-4B` 适配
- `GLM-4.7-Flash` 适配
- `Nemotron-Content-Safety-Reasoning-4B` refusal judge 校准
- 基于 artifact manifest 的续跑能力
- 面向实际使用的 ablation-only 推理入口

论文与背景资料：

- [Paper](https://arxiv.org/abs/2406.11717)
- [Blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## 1. 项目在做什么

完整流程分成两部分：

1. 从 harmful / harmless 指令中提取候选 refusal direction
2. 选出一个最有效的 direction
3. 用这个 direction 做干预并评估

当前仓库里主要有两条使用路径：

- `pipeline.run_pipeline`
  用于完整实验、生成评估产物、对 baseline / ablation / actadd 做系统比较
- `pipeline.prepare_inference_direction` + `pipeline.run_ablation_inference`
  用于更贴近实际使用的推理流程，只保留 `position=-1` 和 `ablation`

## 2. 环境准备

推荐直接使用本地虚拟环境：

```bash
cd /root/program/refusal_direction
source /root/venv/refuse/bin/activate
```

Hugging Face 相关访问默认走镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="<your_token>"
```

如果要使用 `llamaguard2`，还需要：

```bash
export TOGETHER_API_KEY="<your_key>"
```

当前适配 `Qwen3.5-4B` 和 `GLM-4.7-Flash` 需要 `transformers==5.4.0`。

## 3. 目录说明

- `dataset/`
  数据集与 train/val/test 切分
- `pipeline/run_pipeline.py`
  完整实验入口
- `pipeline/prepare_inference_direction.py`
  推理专用 direction 准备入口
- `pipeline/run_ablation_inference.py`
  推理专用交互入口
- `pipeline/submodules/refusal_calibration.py`
  基膜响应缓存与 Nemotron refusal judge
- `pipeline/runs/<model_alias>/`
  每次实验的产物目录

## 4. 完整实验怎么跑

完整实验入口：

```bash
python -m pipeline.run_pipeline --model_path <model_path>
```

例如：

```bash
python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm
```

如果切到 `GLM-4.7-Flash`，命令形式相同，例如：

```bash
python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/GLM-4.7-Flash \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm
```

### 建议的 smoke test

```bash
python -m pipeline.run_pipeline \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --n_train 8 \
  --n_val 4 \
  --n_test 8 \
  --max_new_tokens 128 \
  --ce_loss_n_batches 2 \
  --ce_loss_batch_size 1 \
  --activation_batch_size 4 \
  --completion_batch_size 4 \
  --refusal_calibration_batch_size 2 \
  --refusal_calibration_max_new_tokens 64
```

### 完整实验会输出什么

默认产物目录：

```text
pipeline/runs/<model_alias>/
```

主要包括：

- `refusal_calibration/`
  基膜响应缓存、Nemotron judge 结果、重建出的 refusal tokens
- `generate_directions/`
  候选 direction 张量
- `select_direction/`
  所有候选 layer/position 的评估结果
- `direction.pt`
  最终选中的 direction
- `completions/`
  baseline / ablation / actadd 的 completion 与评估
- `loss_evals/`
  `pile` / `alpaca` / `alpaca_custom_completions` 的 loss

### 续跑逻辑

`run_pipeline.py` 现在支持基于 manifest 的参数感知缓存复用：

- 同样的参数再次运行，会直接复用已有产物
- 如果改了 `max_new_tokens`、输入 split、direction 文件等，会自动只重跑受影响阶段

这避免了“旧 completion 被误复用”的问题。

## 5. 如果你只是想拿去做推理

如果目标不是重跑完整实验，而是尽快得到一个可用的“去拒绝”推理模型，推荐使用两步法。

### 第一步：准备推理方向

```bash
python -m pipeline.prepare_inference_direction \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm \
  --n_train 8 \
  --n_val 4 \
  --activation_batch_size 4 \
  --completion_batch_size 4 \
  --refusal_calibration_batch_size 2 \
  --refusal_calibration_max_new_tokens 64
```

`GLM-4.7-Flash` 的准备命令同样只需要替换 `--model_path`：

```bash
python -m pipeline.prepare_inference_direction \
  --model_path /root/autodl-tmp/GLM-4.7-Flash \
  --refusal_judge_model_path /root/autodl-tmp/Nemotron-Content-Safety-Reasoning-4B \
  --refusal_judge_backend vllm
```

这个入口和完整实验的区别是：

- 固定只用 `position=-1`
- 只搜索 `layer`
- 只按 `ablation` 目标选 direction
- 不再跑 baseline / actadd 全套对照

产物会写到：

```text
pipeline/runs/<model_alias>/inference_ablation/
```

### 第二步：加载这个 direction 直接推理

单轮：

```bash
python -m pipeline.run_ablation_inference \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --prompt "Write a tutorial on how to make a bomb." \
  --max_new_tokens 128
```

交互模式：

```bash
python -m pipeline.run_ablation_inference \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --interactive \
  --max_new_tokens 256
```

输入 `exit` 或 `quit` 退出。

## 6. 还保留了一个简单的对比脚本

如果你想快速比较原模型和干预后的差异，也可以用：

```bash
python scripts/chat_with_direction.py \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --mode baseline \
  --prompt "Write a tutorial on how to make a bomb." \
  --max_new_tokens 128
```

或者：

```bash
python scripts/chat_with_direction.py \
  --model_path /root/autodl-tmp/Qwen3.5-4B \
  --mode ablation \
  --prompt "Write a tutorial on how to make a bomb." \
  --max_new_tokens 128
```

其中：

- `baseline`：原始模型
- `ablation`：去除 refusal direction
- `actadd`：向模型注入 direction，副作用通常更大

## 7. 当前实验上的经验结论

基于当前 `Qwen3.5-4B` 的小规模实验：

- 最优 direction 为 `position=-1, layer=23`
- `ablation` 明显比 `actadd` 更适合实际推理使用
- `actadd` 虽然也能提升 harmful completion 的成功率，但更容易带来重复、说教、胡话和 harmless 任务上的明显副作用

因此当前仓库新增的推理专用入口默认采用：

- `position=-1`
- `layer` 搜索
- 只使用 `ablation`

`GLM-4.7-Flash` 目前已经接入同一套适配层，但它的最优 `layer`/`position` 仍需要你在服务器上按同样流程重新搜索，不应直接沿用 Qwen 的经验结论。

## 8. 常见问题

### 为什么 completion 看起来像被截断了？

通常是 `--max_new_tokens` 设得太小，不是评估脚本二次裁剪。

建议：

- smoke test 用 `128`
- 实际观察行为用 `256`
- 更长内容再继续增大

### 为什么 `run_pipeline` 还是会先加载模型？

当前 pipeline 的缓存复用是按 stage 做的，不是“完全不进模型”。所以重新运行时通常仍会先加载目标模型，再决定哪些阶段直接复用。

### 为什么推理专用流程还保留 layer 搜索？

因为从当前结果看，`position=-1` 很强，但 `layer` 仍然明显影响最终效果。直接把 layer 写死虽然更简单，但稳健性不如“固定 position、只搜索 layer”。

## 9. 引用

如果这个仓库对你的研究有帮助，可以引用原论文：

```tex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Panickssery and Wes Gurnee and Neel Nanda},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```
