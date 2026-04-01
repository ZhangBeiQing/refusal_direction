from transformers import AutoConfig

from pipeline.model_utils.model_base import ModelBase


def _detect_model_family(model_path: str) -> str | None:
    normalized = model_path.lower()

    if "qwen" in normalized:
        return "qwen"
    if "glm" in normalized or "chatglm" in normalized:
        return "glm"
    if "llama-3" in normalized:
        return "llama3"
    if "llama" in normalized:
        return "llama2"
    if "gemma" in normalized:
        return "gemma"
    if "yi" in normalized:
        return "yi"

    try:
        model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type.lower()
    except Exception:
        return None

    if model_type.startswith("qwen"):
        return "qwen"
    if model_type.startswith("glm"):
        return "glm"
    if model_type == "gemma":
        return "gemma"

    return None


def construct_model_base(model_path: str) -> ModelBase:
    family = _detect_model_family(model_path)

    if family == "qwen":
        from pipeline.model_utils.qwen_model import QwenModel

        return QwenModel(model_path)
    if family == "glm":
        from pipeline.model_utils.glm_model import GLMModel

        return GLMModel(model_path)
    if family == "llama3":
        from pipeline.model_utils.llama3_model import Llama3Model

        return Llama3Model(model_path)
    if family == "llama2":
        from pipeline.model_utils.llama2_model import Llama2Model

        return Llama2Model(model_path)
    if family == "gemma":
        from pipeline.model_utils.gemma_model import GemmaModel

        return GemmaModel(model_path)
    if family == "yi":
        from pipeline.model_utils.yi_model import YiModel

        return YiModel(model_path)

    raise ValueError(f"Unknown model family: {model_path}")
