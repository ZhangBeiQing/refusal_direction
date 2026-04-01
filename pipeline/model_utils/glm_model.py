import functools

import torch

from jaxtyping import Float
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List

from pipeline.model_utils.model_base import ENGLISH_OUTPUT_SYSTEM_PROMPT, ModelBase
from pipeline.utils.utils import get_orthogonalized_matrix

GLM_CHAT_TEMPLATE_WITH_SYSTEM = """<|system|>
{system}<|user|>
{instruction}<|assistant|>
"""

GLM_CHAT_TEMPLATE = """<|user|>
{instruction}<|assistant|>
"""

GLM_TEMPLATE_SENTINEL = "<<GLM_INSTRUCTION_SENTINEL>>"
def _resolve_glm_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer
    raise NotImplementedError("Unsupported GLM architecture.")


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


def format_instruction_glm_chat(
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
            formatted_instruction = GLM_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
        else:
            formatted_instruction = GLM_CHAT_TEMPLATE.format(instruction=instruction)

        if output is not None:
            formatted_instruction += output

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    return formatted_instruction


def tokenize_instructions_glm_chat(
    instructions: List[str],
    tokenizer: AutoTokenizer,
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_glm_chat(
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
            format_instruction_glm_chat(
                instruction=instruction,
                tokenizer=tokenizer,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )


def orthogonalize_glm_weights(model, direction: Float[Tensor, "d_model"]):
    text_model = _resolve_glm_text_model(model)

    text_model.embed_tokens.weight.data = get_orthogonalized_matrix(text_model.embed_tokens.weight.data, direction)

    for block in text_model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        if hasattr(block.mlp, "down_proj"):
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T
        elif hasattr(block.mlp, "shared_experts") and hasattr(block.mlp.shared_experts, "down_proj"):
            block.mlp.shared_experts.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.shared_experts.down_proj.weight.data.T,
                direction,
            ).T


def act_add_glm_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    text_model = _resolve_glm_text_model(model)
    mlp = text_model.layers[layer - 1].mlp

    if not hasattr(mlp, "down_proj") or getattr(mlp.down_proj, "bias", None) is None:
        raise NotImplementedError("Weight-space activation addition is not implemented for GLM layers without MLP bias.")

    direction_bias = (coeff * direction).to(dtype=mlp.down_proj.weight.dtype, device=mlp.down_proj.weight.device)
    mlp.down_proj.bias = torch.nn.Parameter(direction_bias)


class GLMModel(ModelBase):
    def __init__(self, model_name_or_path: str):
        self.model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            eos_token = tokenizer.eos_token
            if eos_token is not None:
                tokenizer.pad_token = eos_token
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_glm_chat,
            tokenizer=self.tokenizer,
            system=ENGLISH_OUTPUT_SYSTEM_PROMPT,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = format_instruction_glm_chat(
                instruction=GLM_TEMPLATE_SENTINEL,
                tokenizer=self.tokenizer,
                output=None,
                system=ENGLISH_OUTPUT_SYSTEM_PROMPT,
                include_trailing_whitespace=True,
            )
            suffix = formatted_prompt.split(GLM_TEMPLATE_SENTINEL, 1)[-1]
            return self.tokenizer.encode(suffix, add_special_tokens=False)

        return self.tokenizer.encode(GLM_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        refusal_toks = []
        for candidate in ("I", "As"):
            tok_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if tok_ids:
                refusal_toks.append(tok_ids[0])
        return refusal_toks

    def _get_model_block_modules(self):
        return _resolve_glm_text_model(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_glm_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_glm_weights, direction=direction, coeff=coeff, layer=layer)
