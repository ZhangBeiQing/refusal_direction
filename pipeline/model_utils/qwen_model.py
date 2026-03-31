import functools

import torch

from jaxtyping import Float
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List

from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_TEMPLATE_SENTINEL = "<<QWEN_INSTRUCTION_SENTINEL>>"


def _resolve_qwen_text_config(config):
    return getattr(config, "text_config", config)


def _resolve_qwen_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise NotImplementedError("Unsupported Qwen architecture.")


def _resolve_attn_output_proj(block):
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj
    if hasattr(block, "linear_attn") and hasattr(block.linear_attn, "out_proj"):
        return block.linear_attn.out_proj
    raise NotImplementedError("Could not resolve Qwen attention output projection module.")


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    if not getattr(tokenizer, "chat_template", None):
        raise TypeError("Tokenizer does not expose a chat template.")

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={"enable_thinking": False},
        )


def format_instruction_qwen_chat(
    instruction: str,
    tokenizer: AutoTokenizer,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})
        if output is not None:
            messages.append({"role": "assistant", "content": output})

        formatted_instruction = _apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=output is None,
        )
    else:
        if system is not None:
            formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
        else:
            formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

        if output is not None:
            formatted_instruction += output

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    return formatted_instruction


def tokenize_instructions_qwen_chat(
    instructions: List[str],
    tokenizer: AutoTokenizer,
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                output=output,
                tokenizer=tokenizer,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                tokenizer=tokenizer,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return result


def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    text_model = _resolve_qwen_text_model(model)

    text_model.embed_tokens.weight.data = get_orthogonalized_matrix(text_model.embed_tokens.weight.data, direction)

    for block in text_model.layers:
        attn_out_proj = _resolve_attn_output_proj(block)
        attn_out_proj.weight.data = get_orthogonalized_matrix(attn_out_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T


def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    text_model = _resolve_qwen_text_model(model)
    down_proj = text_model.layers[layer - 1].mlp.down_proj
    bias = getattr(down_proj, "bias", None)

    if bias is None:
        raise NotImplementedError("Weight-space activation addition is not implemented for bias-less Qwen layers.")

    direction_bias = (coeff * direction).to(dtype=down_proj.weight.dtype, device=down_proj.weight.device)
    down_proj.bias = torch.nn.Parameter(direction_bias)


class QwenModel(ModelBase):
    def __init__(self, model_name_or_path: str):
        self.model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.text_config = _resolve_qwen_text_config(self.model_config)
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        model_type = getattr(self.model_config, "model_type", "")

        if model_type != "qwen3_5":
            model_kwargs.update({"use_flash_attn": True})
            if dtype != "auto":
                model_kwargs.update(
                    {
                        "bf16": dtype == torch.bfloat16,
                        "fp16": dtype == torch.float16,
                        "fp32": dtype == torch.float32,
                    }
                )
        model_kwargs["dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
        model.requires_grad_(False)

        text_config = _resolve_qwen_text_config(model.config)
        for attr in ("hidden_size", "num_hidden_layers", "vocab_size"):
            if not hasattr(model.config, attr) and hasattr(text_config, attr):
                setattr(model.config, attr, getattr(text_config, attr))

        return model

    def _load_tokenizer(self, model_path):
        tokenizer_kwargs = {"trust_remote_code": True}
        if getattr(self.model_config, "model_type", "") != "qwen3_5":
            tokenizer_kwargs["use_fast"] = False

        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            eos_token = tokenizer.eos_token
            if eos_token is not None:
                tokenizer.pad_token = eos_token
            else:
                eos_token_id = tokenizer.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_token_id = eos_token_id[0]
                if eos_token_id is not None:
                    tokenizer.pad_token_id = eos_token_id
        if tokenizer.pad_token is None and getattr(self.model_config, "model_type", "") != "qwen3_5":
            tokenizer.pad_token = "<|extra_0|>"
            tokenizer.pad_token_id = tokenizer.eod_id

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = format_instruction_qwen_chat(
                instruction=QWEN_TEMPLATE_SENTINEL,
                tokenizer=self.tokenizer,
                output=None,
                system=None,
                include_trailing_whitespace=True,
            )
            suffix = formatted_prompt.split(QWEN_TEMPLATE_SENTINEL, 1)[-1]
            return self.tokenizer.encode(suffix, add_special_tokens=False)

        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        refusal_toks = []
        for candidate in ("I", "As"):
            tok_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if tok_ids:
                refusal_toks.append(tok_ids[0])
        return refusal_toks

    def _get_model_block_modules(self):
        return _resolve_qwen_text_model(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [
                block_module.self_attn if hasattr(block_module, "self_attn") else block_module.linear_attn
                for block_module in self.model_block_modules
            ]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)
