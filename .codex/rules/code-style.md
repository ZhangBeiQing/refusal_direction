# 代码风格与日志规范

本规则补充 `pre_commit_rule.md`，约束本仓库的编码与日志实现方式。

## 1. 基本代码风格

- Python 统一使用 4 个空格缩进。
- 变量、函数使用 `snake_case`。
- 类名使用 `CapWords` / PascalCase。
- 优先显式导入，禁止 `from x import *`。
- 复杂行为补充类型注解和简短 docstring。

## 2. 目录与职责

| 目录 | 职责 |
|------|------|
| `pipeline/` | 核心管线：模型适配、方向提取、评估、校准 |
| `pipeline/utils/` | 工具函数：hook、线性代数、日志 |
| `dataset/` | 数据加载、raw/processed/splits 管理 |
| `scripts/` | CLI 入口和 demo 调试脚本 |
| `.codex/rules/` | 项目规范文档 |

- `pipeline/` 放可复用的核心逻辑，`scripts/` 放参数解析和一次性调用。
- 新增模型适配放在 `pipeline/model_utils/` 下，并在 `model_factory.py` 注册。

## 3. 设计偏好

- 优先组合而不是深继承。
- 保持实验可复现为最高优先级，不过度工程化。
- 修改 pipeline 子模块时，先确认输出路径和 JSON 字段名不被无意破坏。

## 4. 日志规范

### 4.1 入口

```python
from pipeline.utils.logging import get_logger

logger = get_logger("ComponentName")
```

组件名使用具业务含义的 PascalCase，例如：
- `PrepareInferenceDirection`
- `SelectDirection`
- `RefusalCalibration`
- `GenerateDirections`
- `EvaluateJailbreak`

### 4.2 日志级别

| 级别 | 用途 |
|------|------|
| `DEBUG` | 详细跟踪，仅用于排障或低频路径 |
| `INFO` | 阶段性里程碑、关键状态摘要、输入输出统计 |
| `WARNING` | 可恢复问题、降级、跳过、缓存异常但流程继续 |
| `ERROR` | 当前步骤失败，需要人工关注 |

正常流程用 `INFO`，不要把失败写成 `INFO`。

### 4.3 允许 print 的场景

- CLI 最终结果输出
- demo / 手工调试入口（`scripts/` 下的交互式脚本）
- 终端交互式 prompt（如 `input()`）

### 4.4 示例

```python
logger = get_logger("SelectDirection")

logger.info("开始评估 %d 层候选方向 (kl_threshold=%.2f)", n_layers, kl_threshold)
logger.info("  layer=%2d  refusal=%.4f  kl=%.4f  [kept]", layer, score, kl)
logger.warning("  layer=%2d  kl=%.4f 超过阈值 %.2f，已跳过", layer, kl, kl_threshold)
logger.error("无候选方向通过过滤条件 (prune_layer_percentage=%.1f, kl_threshold=%.2f)",
             prune_layer_percentage, kl_threshold)
```

### 4.5 进度条

- 批量计算（如 token 级别循环）使用 `tqdm`。
- 阶段性里程碑用 `logger.info`，不要和 tqdm 混在一起。
- 如果 tqdm 和 logger 同时输出导致换行错乱，先 `logger.info` 再启动 tqdm。

## 5. 数据格式约定

- 指令样本：`{"instruction": "...", "category": "..."}`
- 补全结果：`{"category": "...", "prompt": "...", "response": "..."}`
- 改动数据管线时，保持上述字段名不变，避免破坏已有评估脚本和历史产物兼容性。

## 6. 迁移原则

- 新代码必须使用 `pipeline.utils.logging`。
- 修改已有模块时顺手把 `print` 迁移到 logger。
- 不要求一次性清干净仓库所有历史 `print`，但主链路新代码必须遵守。
