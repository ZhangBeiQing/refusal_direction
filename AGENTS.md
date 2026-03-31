# AGENTS.md

## 项目定位

这是一个复现论文 *Refusal in Language Models Is Mediated by a Single Direction* 的研究型代码仓库。核心目标是：

1. 从有害/无害指令中提取候选 refusal directions。
2. 选择最有效的 direction。
3. 对 baseline / ablation / activation addition 生成 completion 并做安全评估。
4. 评估 CE loss 与困惑度变化。

仓库包含有害文本、越狱提示和评估产物。修改时默认以“保持实验可复现、尽量不破坏已有产物”为优先。

## 环境要求

- Python `>=3.10`
- 使用本地虚拟环境：`source /root/venv/refuse/bin/activate`
- Hugging Face 访问默认走镜像：`export HF_ENDPOINT="https://hf-mirror.com"`
- 访问 gated Hugging Face 模型与高频 Hub / datasets 请求时需要先导出 `HF_TOKEN`
- 使用 `llamaguard2` 评估时需要 `TOGETHER_API_KEY`
- 运行主流程默认需要 GPU；`vllm` 相关评估和 refusal calibration 也依赖显存

注意：

- 默认先激活本地虚拟环境，不要在这个仓库里重新创建 `venv/`
- 由于网络较差，凡是需要访问 Hugging Face Hub 或 datasets 的命令，默认先执行 `export HF_ENDPOINT="https://hf-mirror.com"`
- 涉及 Hugging Face Hub 或 datasets 的命令，默认同时确保当前 shell 已执行 `export HF_TOKEN="<your_token>"`
- `setup.sh` 会重写根目录 `.env`
- 若未设置 `TOGETHER_API_KEY`，`pipeline.run_pipeline` 会自动移除 `llamaguard2` 评估，只保留本地的 `substring_matching`

## 关键入口

- 主入口：[pipeline/run_pipeline.py](/root/program/refusal_direction/pipeline/run_pipeline.py)
- 配置定义：[pipeline/config.py](/root/program/refusal_direction/pipeline/config.py)
- 数据加载：[dataset/load_dataset.py](/root/program/refusal_direction/dataset/load_dataset.py)

标准运行命令：

```bash
python3 -m pipeline.run_pipeline --model_path <huggingface_model_path>
```

例如：

```bash
python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

若需要 refusal calibration，可额外传：

```bash
--refusal_judge_model_path <judge_model_path>
```

## 目录速览

- [dataset/](/root/program/refusal_direction/dataset)
  - `raw/`: 原始数据
  - `processed/`: 评估使用的数据集
  - `splits/`: `harmful` / `harmless` 的 train/val/test 切分
- [pipeline/](/root/program/refusal_direction/pipeline)
  - `run_pipeline.py`: 串联完整流程
  - `config.py`: 默认实验配置
  - `model_utils/`: 不同模型族的适配层
  - `submodules/`: 方向生成、方向选择、jailbreak 评估、loss 评估、refusal calibration
  - `utils/`: hook 与线性代数工具
  - `runs/`: 已生成的实验产物
- [docs/](/root/program/refusal_direction/docs)
  - 补充设计说明和计划文档

## 主流程与产物

`pipeline.run_pipeline` 的实际执行顺序是：

1. 加载目标模型适配器。
2. 从 `dataset/splits` 采样 harmful/harmless 的 train 和 val。
3. 二选一：
   - 默认使用 refusal score 过滤样本。
   - 若提供 `--refusal_judge_model_path`，先生成缓存响应，再用 judge 模型校准 refusal token 和过滤结果。
4. 在 `generate_directions/` 中生成 `mean_diffs.pt`。
5. 在 `select_direction/` 中评估各层各位置的方向，并输出图和 JSON。
6. 保存最终 `direction.pt` 与 `direction_metadata.json`。
7. 对评估数据集生成 `baseline` / `ablation` / `actadd` completions。
8. 写出 jailbreak / refusal 评估结果。
9. 写出 `loss_evals/*.json`。

默认产物目录：

```text
pipeline/runs/<model_alias>/
```

## 代码结构约定

### 模型适配

模型族支持通过 [pipeline/model_utils/model_factory.py](/root/program/refusal_direction/pipeline/model_utils/model_factory.py) 分发，当前已覆盖：

- Qwen
- Llama 2
- Llama 3
- Gemma
- Yi

如果新增模型族，至少要补齐：

1. chat template / tokenizer 逻辑
2. `eoi_toks`
3. `refusal_toks`
4. block / attention / mlp 模块定位
5. `model_factory.py` 中的分发入口

### Hook 机制

方向消融和 activation addition 统一放在 [pipeline/utils/hook_utils.py](/root/program/refusal_direction/pipeline/utils/hook_utils.py)。

- 优先复用已有 hook
- 不要在多个子模块里复制相同 hook 逻辑
- 修改 hook 时要留意 tuple output 和 tensor output 两种路径

### 数据格式

大多数数据样本遵循如下结构：

```json
{
  "instruction": "...",
  "category": "..."
}
```

completion 结构通常为：

```json
{
  "category": "...",
  "prompt": "...",
  "response": "..."
}
```

改动数据管线时，尽量保持这些字段名不变，避免破坏已有评估脚本与历史产物兼容性。

## 修改建议

- 优先修改 `pipeline/` 与 `dataset/` 下的源代码，不要随意手改 `pipeline/runs/` 中的实验产物。
- 若任务不是“重跑实验并更新产物”，默认不要提交大体积 `.pt`、`.png`、`.json` 结果文件。
- 保持 `Config` 的默认值偏向论文复现；如果要加入更轻量的调试参数，优先通过 CLI override 暴露。
- `run_pipeline.py` 中的路径命名和输出文件名属于外部接口的一部分，修改前先确认是否会影响 README 中的复现实验说明。
- `evaluate_jailbreak.py` 同时包含本地和外部服务评估逻辑，变更时要保留“无 `TOGETHER_API_KEY` 时仍可运行”的退化行为。

## 验证方式

代码改动后，优先做这两类验证。

语法检查：

```bash
python3 -m compileall pipeline dataset
```

小规模 smoke test（需要本地模型/GPU）：

```bash
python3 -m pipeline.run_pipeline \
  --model_path <model_path> \
  --n_train 8 \
  --n_val 4 \
  --n_test 8 \
  --max_new_tokens 128 \
  --ce_loss_n_batches 2 \
  --ce_loss_batch_size 1 \
  --activation_batch_size 1 \
  --completion_batch_size 1 \
  --refusal_calibration_batch_size 2 \
  --refusal_calibration_max_new_tokens 64
```

如果只改了某个子模块，至少确认：

- 导入不报错
- 输出路径仍与现有目录约定一致
- JSON 字段名未被无意改坏

## 代理工作边界

- 这是研究仓库，不要把“工程化重构”放在“保持实验行为一致”之前。
- 默认假设工作区可能已有历史产物，除非任务明确要求，否则不要清理 `pipeline/runs/`。
- 仓库当前没有正式测试套件；没有 GPU 或模型权重时，无法完整验证主流程，需要在结果里明确说明验证边界。
- `run_pipeline.py` 内部使用固定随机种子 `42` 进行采样；若修改采样逻辑，要明确说明是否影响复现性。
